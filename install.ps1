# VigenFlow quick install for Windows. In PowerShell:
#
#   irm https://raw.githubusercontent.com/jia1217/vigenflow/main/install.ps1 | iex
#
# Checks the NPU driver, downloads the latest Windows release, asks which models to download and starts
# vgf-serve.exe in its own window, and OpenWebUI too if you want it (run with uv, so neither Docker nor
# Python has to be installed). Run it again to update VigenFlow, update the weights or add models;
# downloaded weights are kept. The install folder holds what installing a release zip by hand gives:
#   vigenflow_<version>_windows_amd64\   the release package
#   all_model_weights\                   the weights, shared by every version
#   openwebui\                           OpenWebUI, its Python and its data (optional)
#   Start VigenFlow.cmd, Start OpenWebUI.cmd, also in the Start menu under VigenFlow
#
# Unattended runs: set these first, for example $env:VIGENFLOW_MODELS = 'flux2_klein_4B'
#   VIGENFLOW_DIR                  install folder (default: the one used last, else %USERPROFILE%\VigenFlow)
#   VIGENFLOW_MODELS               models to download, comma-separated, or all (default: ask)
#   VIGENFLOW_ACCEPT_KREA_LICENSE  yes accepts Krea 2 Turbo's license (krea2_turbo)
#   VIGENFLOW_WEBUI                1 or 0: start OpenWebUI or not (default: ask)
#
# Written for Windows PowerShell 5.1, which irm | iex runs in by default.

function Install-VigenFlow {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'  # Windows PowerShell downloads many times slower with a progress bar

    $repo = 'jia1217/vigenflow'
    $apiPort = 11283
    $webuiVersion = '0.11.4'  # the OpenWebUI version VigenFlow is tested with
    $allModels = @('flux2_klein_4B', 'z_image_turbo', 'krea2_turbo', 'flux1_schnell')
    $interactive = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected

    function Say([string]$text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
    function Ask([string]$question, [string]$default) {
        if (-not $interactive) { return $default }
        $answer = Read-Host $question
        if ([string]::IsNullOrWhiteSpace($answer)) { return $default }
        return $answer.Trim()
    }
    function Test-PortInUse([int]$port) {
        if (-not (Get-Command Get-NetTCPConnection -ErrorAction SilentlyContinue)) { return $false }
        return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    }
    function Wait-Url([string]$url, [int]$seconds) {
        $deadline = (Get-Date).AddSeconds($seconds)
        while ((Get-Date) -lt $deadline) {
            try {
                Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5 | Out-Null
                return $true
            } catch {
                Start-Sleep -Seconds 2
            }
        }
        return $false
    }
    # cmd reads a batch file in its code page; chcp 65001 on the second line makes it read the rest as
    # UTF-8, so folders with non-English names work. Written without a BOM, which cmd would not skip.
    function Write-Launcher([string]$path, [string[]]$lines) {
        $text = (@('@echo off', 'chcp 65001 >nul') + $lines) -join "`r`n"
        [IO.File]::WriteAllText($path, $text + "`r`n", (New-Object Text.UTF8Encoding $false))
    }
    # Closes a launcher's window and the programs it started from $dir. cmd reads a batch file from disk
    # line by line, so a window still running the old launcher must not see the file rewritten.
    function Stop-Launcher([string]$launcher, [string]$dir) {
        if (Get-Command Get-CimInstance -ErrorAction SilentlyContinue) {
            Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" -ErrorAction SilentlyContinue |
                Where-Object { $_.CommandLine -and $_.CommandLine.IndexOf($launcher, [StringComparison]::OrdinalIgnoreCase) -ge 0 } |
                ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        }
        Get-Process -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -and $_.Path.StartsWith($dir, [StringComparison]::OrdinalIgnoreCase) } |
            Stop-Process -Force -ErrorAction SilentlyContinue
    }
    function Add-StartMenuShortcut([string]$name, [string]$target) {
        try {
            $folder = Join-Path ([Environment]::GetFolderPath('Programs')) 'VigenFlow'
            New-Item -ItemType Directory -Force -Path $folder | Out-Null
            $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $folder "$name.lnk"))
            $shortcut.TargetPath = $target
            $shortcut.WorkingDirectory = Split-Path -Parent $target
            $shortcut.Save()
        } catch {
            Write-Host "  (no Start menu shortcut: $($_.Exception.Message))"
        }
    }

    if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
        throw 'This installer is for 64-bit Windows. On Ubuntu, use docker/install.sh (see the README).'
    }
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    # --- NPU driver -------------------------------------------------------------------------------------
    if (Get-Command Get-PnpDevice -ErrorAction SilentlyContinue) {
        $devices = @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue)
        $npu = @($devices | Where-Object { $_.InstanceId -like 'PCI\VEN_1022&DEV_17F0*' })
        if (-not $npu) {
            if ($devices | Where-Object { $_.InstanceId -like 'PCI\VEN_1022&DEV_1502*' }) {
                throw 'This PC has a Ryzen 7040/8040-series NPU. VigenFlow needs a Ryzen AI 300-series (XDNA 2) NPU.'
            }
            throw 'No Ryzen AI 300-series NPU was found. VigenFlow needs one, with the NPU turned on in the BIOS.'
        }
        if ($npu[0].Status -ne 'OK') {
            throw ("The NPU has no working driver (Device Manager status: $($npu[0].Status)). Install the NPU driver from " +
                   'https://www.amd.com/en/support/download/drivers.html, restart, and run this installer again.')
        }
        Say 'NPU found, with its driver working'
    } else {
        Say 'Cannot check the NPU driver on this Windows version; continuing'
    }

    # --- Install folder ---------------------------------------------------------------------------------
    $stateFile = Join-Path (Join-Path $env:LOCALAPPDATA 'VigenFlow') 'install_folder.txt'
    $defaultRoot = Join-Path $env:USERPROFILE 'VigenFlow'
    if (Test-Path -LiteralPath $stateFile) { $defaultRoot = (Get-Content -LiteralPath $stateFile -Raw).Trim() }
    $root = $env:VIGENFLOW_DIR
    if (-not $root) {
        Write-Host "`nThe install folder needs room for the model weights: 8 to 65 GB, depending on the models."
        $root = Ask "Install folder [$defaultRoot]" $defaultRoot
    }
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    $root = (Resolve-Path -LiteralPath $root).Path
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $stateFile) | Out-Null
    Set-Content -LiteralPath $stateFile -Value $root
    $drive = (Get-Item -LiteralPath $root).PSDrive
    if ($drive -and $drive.Free) { Write-Host "  $root ($([math]::Floor($drive.Free / 1GB)) GB free)" } else { Write-Host "  $root" }
    $startServer = Join-Path $root 'Start VigenFlow.cmd'
    $startWebui = Join-Path $root 'Start OpenWebUI.cmd'

    # --- Latest release ---------------------------------------------------------------------------------
    Say 'Finding the latest VigenFlow release'
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest" -Headers @{ 'User-Agent' = 'vigenflow-install' }
    $asset = @($release.assets | Where-Object { $_.name -like '*_windows_amd64.zip' })[0]
    if (-not $asset) { throw "Release $($release.tag_name) has no Windows package." }
    $packageName = [IO.Path]::GetFileNameWithoutExtension($asset.name)
    $package = Join-Path $root $packageName
    $serverDir = Join-Path $package 'server_running'
    $server = Join-Path $serverDir 'vgf-serve.exe'

    # A vgf-serve.exe started from this folder holds the port and its files; stop it, and its window.
    Get-ChildItem -LiteralPath $root -Directory -Filter 'vigenflow_*_windows_amd64' -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Launcher $startServer $_.FullName }
    if (Test-PortInUse $apiPort) {
        throw "Port $apiPort is in use, probably by a vgf-serve started from another folder. Close it and run this again."
    }

    if (Test-Path -LiteralPath $server) {
        Write-Host "  $packageName ($($release.tag_name)) is already installed"
    } else {
        $zip = Join-Path $root $asset.name
        $staging = Join-Path $root "$packageName.partial"
        Say "Downloading $($asset.name) ($([math]::Round($asset.size / 1MB)) MB)"
        & curl.exe -fL --retry 3 -o $zip $asset.browser_download_url
        if ($LASTEXITCODE -ne 0) { throw "The download failed (curl exit code $LASTEXITCODE)." }
        if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
        Expand-Archive -LiteralPath $zip -DestinationPath $staging
        if (Test-Path -LiteralPath $package) { Remove-Item -LiteralPath $package -Recurse -Force }
        Move-Item -LiteralPath (Join-Path $staging $packageName) -Destination $package
        Remove-Item -LiteralPath $staging, $zip -Recurse -Force
        Write-Host "  Installed $package"
    }

    # --- Models -----------------------------------------------------------------------------------------
    $weights = Join-Path $root 'all_model_weights'
    $models = $env:VIGENFLOW_MODELS
    if (-not $models) {
        Write-Host @'

Which models do you want to download?
  1) FLUX.2-klein-4B    8 GB   fastest (about 9 s per image), also edits images
  2) Z-Image-Turbo     12 GB
  3) Krea 2 Turbo      19 GB   has its own license, which you are asked to accept
  4) FLUX.1-schnell    27 GB
  a) all four          65 GB
Models you already have are only updated.
'@
        $chosen = @()
        foreach ($choice in ((Ask 'Type the numbers separated by spaces [1]' '1') -split '[\s,]+')) {
            if ($choice -in @('a', 'all')) { $chosen = @('all'); break }
            if ($choice -match '^[1-4]$') { $chosen += $allModels[[int]$choice - 1] }
            elseif ($choice) { throw "Unknown choice: $choice" }
        }
        $models = @($chosen | Select-Object -Unique) -join ','
    }
    if ($models -ne 'all') {
        foreach ($m in ($models -split ',')) {
            if ($m -notin $allModels) { throw "Unknown model $m (choose from: $($allModels -join ', '))" }
        }
    }
    # Asked now, so nothing waits for an answer after the long downloads.
    $webui = $env:VIGENFLOW_WEBUI
    if (-not $webui) {
        $webui = '1'
        if ((Ask 'Also start OpenWebUI, the chat page you use VigenFlow in (about 2 GB the first time)? [Y/n]' 'y') -match '^[Nn]') { $webui = '0' }
    }

    Say 'Downloading the model weights (the first time takes a while; an interrupted download resumes)'
    $prepareArgs = @()
    if ($models -ne 'all') { $prepareArgs += @('--only', $models) }
    if ($env:VIGENFLOW_ACCEPT_KREA_LICENSE -eq 'yes') { $prepareArgs += '--accept_license' }
    & (Join-Path (Join-Path $package 'prepare') 'prepare_base.exe') @prepareArgs
    $prepareOk = $LASTEXITCODE -eq 0

    # A model folder exists only once its download is complete (prepare unpacks into <id>.partial first).
    $installed = @()
    if (Test-Path -LiteralPath $weights) {
        $folders = @(Get-ChildItem -LiteralPath $weights -Directory | ForEach-Object { $_.Name })
        $installed = @($allModels | Where-Object { $_ -in $folders })
    }
    if (-not $installed) { throw 'No model weights were downloaded; see the messages above.' }
    if (-not $prepareOk) { Say 'Some models were not downloaded (see above). Starting with the ones that are here.' }
    # z-image-turbo is vgf-serve's own default; otherwise the first model that is here.
    $defaultModel = $installed[0] -replace '_', '-'
    if ('z_image_turbo' -in $installed) { $defaultModel = 'z-image-turbo' }

    # --- Start VigenFlow --------------------------------------------------------------------------------
    Write-Launcher $startServer @(
        'title VigenFlow server'
        "cd /d `"$serverDir`""
        "vgf-serve.exe --model $defaultModel"
        'if errorlevel 1 pause'
    )
    Add-StartMenuShortcut 'VigenFlow server' $startServer
    Say "Starting VigenFlow in its own window (default model: $defaultModel)"
    Start-Process -FilePath $startServer -WorkingDirectory $root
    if (-not (Wait-Url "http://127.0.0.1:$apiPort/v1/models" 60)) {
        throw 'VigenFlow did not start; see the messages in its window.'
    }

    # --- OpenWebUI --------------------------------------------------------------------------------------
    $webuiUrl = ''
    if ($webui -eq '1') {
        $webuiDir = Join-Path $root 'openwebui'
        $uvDir = Join-Path $webuiDir 'uv'
        $uv = Join-Path $uvDir 'uv.exe'
        $dataDir = Join-Path $webuiDir 'data'
        New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
        if (-not (Test-Path -LiteralPath $uv)) {
            Say 'Downloading uv, which runs OpenWebUI with its own Python'
            $uvZip = Join-Path $webuiDir 'uv.zip'
            & curl.exe -fL --retry 3 -o $uvZip 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
            if ($LASTEXITCODE -ne 0) { throw "The uv download failed (curl exit code $LASTEXITCODE)." }
            Expand-Archive -LiteralPath $uvZip -DestinationPath $uvDir -Force
            Remove-Item -LiteralPath $uvZip -Force
        }
        # An OpenWebUI started from this folder keeps running: it needs no restart for new models.
        $portFile = Join-Path $webuiDir 'port.txt'
        $running = @(Get-Process -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -and $_.Path.StartsWith($webuiDir, [StringComparison]::OrdinalIgnoreCase) })
        if ($running -and (Test-Path -LiteralPath $portFile)) {
            $webuiUrl = "http://localhost:$((Get-Content -LiteralPath $portFile -Raw).Trim())"
            Say "OpenWebUI is already running at $webuiUrl"
        }
    }
    if ($webui -eq '1' -and -not $webuiUrl) {
        Stop-Launcher $startWebui $webuiDir
        $webuiPort = 8080
        while (Test-PortInUse $webuiPort) { $webuiPort++ }
        Set-Content -LiteralPath $portFile -Value $webuiPort
        # The connection settings apply on OpenWebUI's first start; later it keeps them in its data folder.
        # It keeps its secret key in the folder it starts in, so it starts in the data folder.
        Write-Launcher $startWebui @(
            'title OpenWebUI for VigenFlow'
            "set `"UV_CACHE_DIR=$(Join-Path $webuiDir 'cache')`""
            "set `"UV_PYTHON_INSTALL_DIR=$(Join-Path $webuiDir 'python')`""
            "set `"DATA_DIR=$dataDir`""
            'set "ENABLE_OLLAMA_API=False"'
            "set `"OPENAI_API_BASE_URL=http://127.0.0.1:$apiPort/v1`""
            'set "OPENAI_API_KEY=vigenflow"'
            'set "ENABLE_IMAGE_GENERATION=True"'
            'set "IMAGE_GENERATION_ENGINE=openai"'
            "set `"IMAGES_OPENAI_API_BASE_URL=http://127.0.0.1:$apiPort/v1`""
            'set "IMAGES_OPENAI_API_KEY=vigenflow"'
            "cd /d `"$dataDir`""
            "`"$uv`" tool run --python 3.11 open-webui@$webuiVersion serve --host 127.0.0.1 --port $webuiPort"
            'if errorlevel 1 pause'
        )
        Add-StartMenuShortcut 'OpenWebUI for VigenFlow' $startWebui
        Say "Starting OpenWebUI on port $webuiPort in its own window"
        Write-Host '  The first start downloads OpenWebUI and its Python, which takes several minutes. This window'
        Write-Host '  waits until it is ready (up to 20 minutes), then opens it in your browser.'
        Start-Process -FilePath $startWebui -WorkingDirectory $root
        $webuiUrl = "http://localhost:$webuiPort"
        if (Wait-Url "http://127.0.0.1:$webuiPort/health" 1200) {
            Start-Process $webuiUrl
        } else {
            Say "OpenWebUI is still starting; open $webuiUrl once its window shows that it is running"
        }
    }

    # --- Done -------------------------------------------------------------------------------------------
    Say 'VigenFlow is running'
    Write-Host "  Models downloaded: $($installed -join ', ')"
    if ($webuiUrl) {
        Write-Host "  1. Open $webuiUrl and create an account (the first one is the admin)."
        Write-Host "  2. Start a new chat and pick a VigenFlow model you downloaded, for example $defaultModel."
        Write-Host '  3. Turn on Image under Integrations in the message box, and type what you want to see.'
    } else {
        Write-Host "  Connect your OpenWebUI to http://127.0.0.1:$apiPort/v1 as in the VigenFlow README."
    }
    Write-Host ''
    Write-Host '  After a restart, start VigenFlow (and OpenWebUI) again from the Start menu, under VigenFlow,'
    Write-Host "  or with the .cmd files in $root."
    Write-Host '  Update, or add models: run this installer again.'
}

try {
    Install-VigenFlow
} catch {
    Write-Host "`nError: $($_.Exception.Message)" -ForegroundColor Red
}
