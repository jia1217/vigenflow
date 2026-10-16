# Checks the Windows launchers ..\exe_models_win\<model>\run.exe and the host DLLs in ..\lib, without
# the NPU: each launcher imports only KERNEL32 and its host_<model>.dll, each host DLL exports
# <model>_host_main, and every DLL they need (recursively) is in ..\lib or is part of Windows, so a
# package with server_running\lib runs on a PC without the VC++ redistributable. On a CPU with AVX-512
# (the host DLLs are built with /arch:AVX512) it also runs each launcher with --help.
#   powershell -ExecutionPolicy Bypass -File check_windows.ps1
[CmdletBinding()]
param([string[]]$Models = @('flux1_schnell', 'flux2_klein_4B', 'flux2_klein_4B_edit', 'krea2_turbo', 'z_image_turbo'))

$ErrorActionPreference = 'Stop'
$srcRoot = Split-Path -Parent $PSScriptRoot
$libDir = Join-Path $srcRoot 'lib'
$launcherDir = Join-Path $srcRoot 'exe_models_win'
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$dumpbin = & $vswhere -latest -products * -find 'VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe' | Select-Object -First 1
if (-not $dumpbin) { throw 'dumpbin.exe not found: install the Visual Studio C++ tools' }
$system32 = Join-Path $env:SystemRoot 'System32'
# Redistributable DLLs: in System32 only where the VC++ redistributable is installed, so lib\ must have them.
$redistributable = '^(msvcp140.*|vcruntime140.*|concrt140|vccorlib140|libomp140.*)\.dll$'
$failures = [System.Collections.Generic.List[string]]::new()

function Get-Imports([string]$file) {
    & $dumpbin /dependents $file | Where-Object { $_ -match '^\s{4}(\S+\.dll)\s*$' } | ForEach-Object { $Matches[1] }
}

$checked = @{}
function Test-Dependencies([string]$file) {
    foreach ($dll in Get-Imports $file) {
        $key = $dll.ToLowerInvariant()
        if ($checked.ContainsKey($key)) { continue }
        $checked[$key] = $true
        $inLib = Join-Path $libDir $dll
        if (Test-Path -LiteralPath $inLib) {
            Test-Dependencies $inLib
        } elseif ($dll -match '^(api|ext)-ms-' -or ($dll -notmatch $redistributable -and (Test-Path -LiteralPath (Join-Path $system32 $dll)))) {
            # part of Windows
        } else {
            $failures.Add("$(Split-Path -Leaf $file) needs $dll, which is neither in $libDir nor part of Windows")
        }
    }
}

Add-Type -Namespace VgfCheck -Name Cpu -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint feature);'
$avx512 = [VgfCheck.Cpu]::IsProcessorFeaturePresent(41)  # PF_AVX512F_INSTRUCTIONS_AVAILABLE

foreach ($model in $Models) {
    $exe = Join-Path $launcherDir "$model\run.exe"
    $dll = Join-Path $libDir "host_$model.dll"
    foreach ($file in $exe, $dll) {
        if (-not (Test-Path -LiteralPath $file)) { $failures.Add("missing $file"); continue }
    }
    if (-not ((Test-Path -LiteralPath $exe) -and (Test-Path -LiteralPath $dll))) { continue }

    $imports = @(Get-Imports $exe | ForEach-Object { $_.ToLowerInvariant() } | Sort-Object)
    $expected = @("host_$($model.ToLowerInvariant()).dll", 'kernel32.dll')
    if (Compare-Object $imports $expected) {
        $failures.Add("$model run.exe imports $($imports -join ', '); expected $($expected -join ', ')")
    }
    $entry = "${model}_host_main"
    if (-not (& $dumpbin /exports $dll | Where-Object { $_ -match "\s$([regex]::Escape($entry))$" })) {
        $failures.Add("host_$model.dll does not export $entry")
    }
    Test-Dependencies $dll

    if ($avx512) {
        $help = & $exe --help 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0 -or $help -notmatch '--weights_path') {
            $failures.Add("$model run.exe --help exited with $LASTEXITCODE`n$help")
        }
    }
    Write-Host "checked $model"
}
if (-not $avx512) {
    Write-Host 'This CPU has no AVX-512, so the launchers were not run (the host DLLs need it); the import, export and dependency checks ran.'
}
if ($failures.Count) {
    $failures | ForEach-Object { Write-Host "FAILED: $_" }
    exit 1
}
Write-Host "PASS: $($Models.Count) launchers and host DLLs"
