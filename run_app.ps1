Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RuntimeDir = Join-Path $ProjectRoot ".runtime"
$BackendOut = Join-Path $RuntimeDir "backend.out.log"
$BackendErr = Join-Path $RuntimeDir "backend.err.log"
$BackendUrl = "http://127.0.0.1:8000"
$FrontendUrl = "http://127.0.0.1:8501"
$ThreeClassCheckpoint = Join-Path $ProjectRoot "ml\outputs\models\dermascan-acne-eczema-psoriasis-custom-cnn.pt"
$BackendProcess = $null

function Test-Url {
    param([string]$Url)
    try {
        Invoke-RestMethod -Uri $Url -TimeoutSec 2 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Start-Backend {
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Python
    $startInfo.WorkingDirectory = $ProjectRoot
    $startInfo.Arguments = "-m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000"
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $null = $process.Start()

    $outStream = [System.IO.File]::Open($BackendOut, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    $errStream = [System.IO.File]::Open($BackendErr, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    $process.StandardOutput.BaseStream.CopyToAsync($outStream) | Out-Null
    $process.StandardError.BaseStream.CopyToAsync($errStream) | Out-Null

    return $process
}

if (-not (Test-Path $Python)) {
    Write-Host "Python virtual environment was not found at $Python" -ForegroundColor Red
    Write-Host "Create it first with: py -3.11 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}

New-Item -ItemType Directory -Force $RuntimeDir | Out-Null
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($env:DERMASCAN_INFERENCE_MODE)) {
    if (Test-Path $ThreeClassCheckpoint) {
        $env:DERMASCAN_INFERENCE_MODE = "checkpoint"
        $env:DERMASCAN_MODEL_PATH = $ThreeClassCheckpoint
        $env:DERMASCAN_MODEL_NAME = "dermascan-acne-eczema-psoriasis-custom-cnn"
        $env:DERMASCAN_MODEL_VERSION = "custom-cnn-acne-eczema-psoriasis-v0.1"
        $env:DERMASCAN_ALLOW_SMOKE_MODEL = "false"
        Write-Host "Using trained three-class checkpoint: $ThreeClassCheckpoint" -ForegroundColor Green
    }
    else {
        $env:DERMASCAN_INFERENCE_MODE = "placeholder"
        Write-Host "Three-class checkpoint not found; using placeholder mode." -ForegroundColor Yellow
    }
}

try {
    if (Test-Url "$BackendUrl/health") {
        Write-Host "Backend already running at $BackendUrl" -ForegroundColor Yellow
    }
    else {
        Write-Host "Starting backend at $BackendUrl ..."
        $BackendProcess = Start-Backend

        $ready = $false
        for ($attempt = 1; $attempt -le 45; $attempt++) {
            Start-Sleep -Seconds 1
            if (Test-Url "$BackendUrl/health") {
                $ready = $true
                break
            }
            if ($BackendProcess.HasExited) {
                Write-Host "Backend exited before it became ready." -ForegroundColor Red
                Write-Host "Backend error log: $BackendErr"
                Get-Content $BackendErr -Tail 40
                exit 1
            }
        }

        if (-not $ready) {
            Write-Host "Backend did not become ready within 45 seconds." -ForegroundColor Red
            Write-Host "Backend error log: $BackendErr"
            Get-Content $BackendErr -Tail 40
            exit 1
        }

        Write-Host "Backend ready." -ForegroundColor Green
    }

    $env:DERMASCAN_API_URL = $BackendUrl
    Write-Host "Starting Streamlit at $FrontendUrl ..."
    Write-Host "Open $FrontendUrl in your browser. Press Ctrl+C here to stop the app."
    & $Python -m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501
}
finally {
    if ($null -ne $BackendProcess -and -not $BackendProcess.HasExited) {
        Write-Host "Stopping backend ..."
        $BackendProcess.Kill()
        $BackendProcess.WaitForExit()
    }
}
