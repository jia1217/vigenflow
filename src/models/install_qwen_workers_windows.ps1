# Stage native Windows Qwen hosts and their runtime DLLs in src/exe_models_win.
# QwenSourceRoot contains NPU_qwen_turbo and NPU_qwen_image with configured CMake builds.
[CmdletBinding()]
param(
    [string]$QwenSourceRoot = (Join-Path $PSScriptRoot '..\..\..\NPU_qwen'),
    [string]$OutDir = (Join-Path $PSScriptRoot '..\exe_models_win'),
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$libDir = Join-Path $PSScriptRoot '..\lib'
$runtimeNames = 'msvcp140.dll', 'msvcp140_atomic_wait.dll', 'vcruntime140.dll', 'vcruntime140_1.dll'
$projects = [ordered]@{
    qwen_image_2_1_turbo = 'NPU_qwen_turbo'
    qwen_image_2_1_turbo_edit = 'NPU_qwen_image'
}

# Validate both inputs before replacing any installed worker. The project-root run.exe is ELF.
foreach ($model in $projects.Keys) {
    $build = Join-Path $QwenSourceRoot ($projects[$model] + '\build-windows')
    if (-not $SkipBuild) {
        & cmake --build $build --config Release --parallel 4
        if ($LASTEXITCODE -ne 0) { throw "Qwen build failed: $build" }
    }
    $exe = Join-Path $build 'Release\run.exe'
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw "Missing $exe. Configure and build the native Windows host first."
    }
    $bytes = [IO.File]::ReadAllBytes($exe)
    if ($bytes.Length -lt 64 -or $bytes[0] -ne 0x4d -or $bytes[1] -ne 0x5a) {
        throw "Expected a Windows PE executable: $exe"
    }
    $pe = [BitConverter]::ToInt32($bytes, 0x3c)
    if ($pe -lt 0 -or $pe + 6 -gt $bytes.Length -or
        [BitConverter]::ToUInt32($bytes, $pe) -ne 0x4550 -or
        [BitConverter]::ToUInt16($bytes, $pe + 4) -ne 0x8664) {
        throw "Expected a Windows x64 executable: $exe"
    }
}
foreach ($name in $runtimeNames) {
    if (-not (Test-Path -LiteralPath (Join-Path $libDir $name) -PathType Leaf)) {
        throw "Missing VC++ runtime: $libDir\$name"
    }
}

foreach ($model in $projects.Keys) {
    $release = Join-Path $QwenSourceRoot ($projects[$model] + '\build-windows\Release')
    $target = Join-Path $OutDir $model
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Copy-Item -LiteralPath (Join-Path $release 'run.exe') -Destination $target
    Get-ChildItem -LiteralPath $release -File -Filter '*.dll' | Copy-Item -Destination $target
    foreach ($name in $runtimeNames) {
        Copy-Item -LiteralPath (Join-Path $libDir $name) -Destination $target
    }
    & (Join-Path $target 'run.exe') --help | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Installed worker failed to start: $target\run.exe" }
    Write-Host "Installed $target\run.exe and its runtime DLLs"
}
