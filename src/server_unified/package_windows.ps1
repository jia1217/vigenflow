# Builds vgf-serve.exe, prepare_base.exe and the model launchers and creates the Windows release zip,
# the counterpart of package_ubuntu.sh:
#
#   vigenflow_<Version>_windows_amd64\
#     server_running\vgf-serve.exe                 with libpng16.dll, zlib1.dll and the VC++ runtime DLLs
#     server_running\exe_models\<model>\run.exe   launchers from src\exe_models_win
#     server_running\lib\                          host_<model>.dll and the DLLs they load, from src\lib
#     server_running\images\
#     all_npu_files\<model>\                       from src\npu_files
#     prepare\prepare_base.exe, prepare_models.jsonc
#     LICENSE, MODEL_LICENSES.md, THIRD_PARTY_NOTICES.md
#
# The model weights are not packaged: prepare\prepare_base.exe downloads them into all_model_weights\ next
# to the extracted package folder, where vgf-serve reads them.
#   .\package_windows.ps1 -Version 0.2.0-beta.3
#   .\package_windows.ps1 -SkipBuild        package the binaries that are already built
[CmdletBinding()]
param(
    [string]$Version = '0.2.0-beta.2',
    [string]$OutRoot = (Join-Path $PSScriptRoot 'release'),
    [string]$PackageName,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$srcRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $srcRoot
$prepareDir = Join-Path $srcRoot 'prepare_weights'
$libDir = Join-Path $srcRoot 'lib'
$launcherDir = Join-Path $srcRoot 'exe_models_win'
$npuFilesDir = Join-Path $srcRoot 'npu_files'
if (-not $PackageName) { $PackageName = "vigenflow_${Version}_windows_amd64" }
$models = 'flux1_schnell', 'flux2_klein_4B', 'flux2_klein_4B_edit', 'krea2_turbo', 'z_image_turbo'

function Invoke-Build([string]$script, [string[]]$arguments = @()) {
    # Build tools write notes to stderr (vcvars64 too); with 'Stop', Windows PowerShell would treat a
    # redirected stderr line as a failure. The exit code decides.
    $ErrorActionPreference = 'Continue'
    & cmd.exe /c "`"$script`" $($arguments -join ' ')" 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$script failed with exit code $LASTEXITCODE" }
}

if (-not $SkipBuild) {
    Invoke-Build (Join-Path $PSScriptRoot 'build_windows.bat')
    Invoke-Build (Join-Path $prepareDir 'build_windows.bat')
    Invoke-Build (Join-Path $srcRoot 'models\build_windows.bat')
}

$serverBin = Join-Path $PSScriptRoot 'vigenflow_win\vgf-serve.exe'
$prepareBin = Join-Path $prepareDir 'vigenflow_win\prepare_base.exe'
$required = @($serverBin, $prepareBin, (Join-Path $prepareDir 'prepare_models.jsonc'))
foreach ($model in $models) {
    $required += Join-Path $launcherDir "$model\run.exe"
    $required += Join-Path $libDir "host_$model.dll"
}
# vgf-serve.exe and the host DLLs need the VC++ runtime unless the redistributable is installed.
$vcRuntime = 'msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'
$required += $vcRuntime | ForEach-Object { Join-Path $libDir $_ }
foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath $file)) {
        throw "Missing $file (build it, or run without -SkipBuild; see docs\README_src.md)"
    }
}

New-Item -ItemType Directory -Force $OutRoot | Out-Null
$OutRoot = (Resolve-Path -LiteralPath $OutRoot).Path
$stage = Join-Path $OutRoot $PackageName
$zip = "$stage.zip"
foreach ($old in $stage, $zip) {
    if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Recurse -Force }
}
$server = Join-Path $stage 'server_running'
New-Item -ItemType Directory -Force (Join-Path $server 'images'), (Join-Path $server 'exe_models'), `
    (Join-Path $server 'lib'), (Join-Path $stage 'prepare') | Out-Null

Copy-Item -LiteralPath $serverBin -Destination $server
Get-ChildItem -LiteralPath (Split-Path -Parent $serverBin) -Filter *.dll | Copy-Item -Destination $server
$vcRuntime | ForEach-Object { Copy-Item -LiteralPath (Join-Path $libDir $_) -Destination $server }
foreach ($model in $models) {
    $target = Join-Path $server "exe_models\$model"
    New-Item -ItemType Directory -Force $target | Out-Null
    Copy-Item -LiteralPath (Join-Path $launcherDir "$model\run.exe") -Destination $target
}
# Every DLL in src\lib (the host DLLs and their runtime DLLs); the Ubuntu .so files and the import
# libraries are only for building.
Get-ChildItem -LiteralPath $libDir -Filter *.dll | Copy-Item -Destination (Join-Path $server 'lib')
Copy-Item -LiteralPath $npuFilesDir -Destination (Join-Path $stage 'all_npu_files') -Recurse
Copy-Item -LiteralPath $prepareBin, (Join-Path $prepareDir 'prepare_models.jsonc') -Destination (Join-Path $stage 'prepare')
foreach ($doc in 'LICENSE', 'MODEL_LICENSES.md', 'THIRD_PARTY_NOTICES.md') {
    Copy-Item -LiteralPath (Join-Path $repoRoot $doc) -Destination $stage
}
Copy-Item -LiteralPath (Join-Path $repoRoot 'model_licenses') -Destination $stage -Recurse

# Entry names use / as the zip format requires (Windows PowerShell's ZipFile.CreateFromDirectory writes \,
# which other unzip tools take as part of the file name), and empty folders such as images\ are kept.
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::Open($zip, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    $base = Split-Path -Parent $stage
    foreach ($item in Get-ChildItem -LiteralPath $stage -Recurse -Force | Sort-Object FullName) {
        $name = $item.FullName.Substring($base.Length + 1).Replace('\', '/')
        if ($item.PSIsContainer) {
            if (-not (Get-ChildItem -LiteralPath $item.FullName -Force | Select-Object -First 1)) {
                [void]$archive.CreateEntry("$name/")
            }
        } else {
            [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $item.FullName, $name, [System.IO.Compression.CompressionLevel]::Optimal)
        }
    }
} finally {
    $archive.Dispose()
}

Write-Host "Created $zip"
Write-Host "Release folder: $stage"
