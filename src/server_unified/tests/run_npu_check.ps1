# Builds the Windows release package, unpacks it next to an existing all_model_weights folder, runs
# npu_check.py against it on the real NPU and removes the unpacked package again. Used by the
# npu-test.yml workflow on the self-hosted runner; it also runs by hand:
#   powershell -ExecutionPolicy Bypass -File tests\run_npu_check.ps1 -WeightsRoot C:\vgf\all_model_weights
#   ... -SkipBuild -Models flux2-klein-4B,flux2-klein-4B-edit -Out C:\temp\npu_check
# The parent folder of -WeightsRoot must be writable; only the package folder this script creates there
# is deleted, and the weights are only read.
[CmdletBinding()]
param(
    [string]$WeightsRoot = $env:VGF_WEIGHTS_ROOT,
    [string]$Models,
    [string]$Out = (Join-Path ([System.IO.Path]::GetTempPath()) 'vgf_npu_check'),
    [int]$Port = 11290,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
if (-not $WeightsRoot) { throw 'Pass -WeightsRoot or set VGF_WEIGHTS_ROOT to the all_model_weights folder' }
$WeightsRoot = (Resolve-Path -LiteralPath $WeightsRoot).Path
if ((Split-Path -Leaf $WeightsRoot) -ne 'all_model_weights') {
    throw "$WeightsRoot must be a folder named all_model_weights: vgf-serve looks for that name next to the package"
}
$serverDir = Split-Path -Parent $PSScriptRoot
$version = 'npucheck-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
$packageName = "vigenflow_${version}_windows_amd64"
$buildRoot = Join-Path ([System.IO.Path]::GetTempPath()) "vgf_npu_package_$version"
$unpacked = Join-Path (Split-Path -Parent $WeightsRoot) $packageName
if (Test-Path -LiteralPath $unpacked) { throw "$unpacked already exists" }

$package = @{ Version = $version; OutRoot = $buildRoot; SkipBuild = [bool]$SkipBuild }
& (Join-Path $serverDir 'package_windows.ps1') @package
Add-Type -AssemblyName System.IO.Compression.FileSystem
$created = $false
try {
    [System.IO.Compression.ZipFile]::ExtractToDirectory((Join-Path $buildRoot "$packageName.zip"), (Split-Path -Parent $WeightsRoot))
    $created = $true
    Write-Host "Unpacked $unpacked next to $WeightsRoot"
    $check = @((Join-Path $PSScriptRoot 'npu_check.py'), $unpacked, '--port', $Port, '--out', $Out)
    if ($Models) { $check += @('--models', $Models) }
    $ErrorActionPreference = 'Continue'  # vgf-serve and the workers log to stderr
    & python @check
    $result = $LASTEXITCODE
} finally {
    $ErrorActionPreference = 'Stop'
    # The unpacked folder comes straight from the zip, so it holds no links into the weights.
    if ($created) { Remove-Item -LiteralPath $unpacked -Recurse -Force }
    if (Test-Path -LiteralPath $buildRoot) { Remove-Item -LiteralPath $buildRoot -Recurse -Force }
}
Write-Host "Images, server log and summary.json: $Out"
exit $result
