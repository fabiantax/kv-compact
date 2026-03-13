# Profile kv-compact on Windows using ETW (Event Tracing for Windows)
# Requires: Windows Performance Toolkit (part of Windows SDK)

param(
    [string]$OutputDir = ".\profiling",
    [int]$DurationSeconds = 30,
    [string]$ExePath = "C:\Users\fabia\Projects\kv-compact\build-native\Release\llama-kv-compact.exe"
)

Write-Host "=== KV Compact Profiling Script ===" -ForegroundColor Cyan
Write-Host ""

# Check if WPT is installed
$wpr = Get-Command wpr -ErrorAction SilentlyContinue
if (-not $wpr) {
    Write-Host "ERROR: Windows Performance Recorder (WPR) not found." -ForegroundColor Red
    Write-Host "Install Windows Performance Toolkit from:" -ForegroundColor Yellow
    Write-Host "https://learn.microsoft.com/en-us/windows-hardware/test/wpt/"
    exit 1
}

# Create output directory
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$traceFile = Join-Path $OutputDir "kv-compact_$timestamp.etl"

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Executable: $ExePath"
Write-Host "  Duration: $DurationSeconds seconds"
Write-Host "  Output: $traceFile"
Write-Host ""

# Check if executable exists
if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: Executable not found: $ExePath" -ForegroundColor Red
    exit 1
}

# Set PATH for DLLs
$env:PATH = "C:\Users\fabia\Projects\kv-compact\build-native\Release;C:\Users\fabia\Projects\kv-compact\build-native\bin\Release;" + $env:PATH

Write-Host "Step 1: Starting performance capture..." -ForegroundColor Yellow
wpr -start CPUProfile -start MemInfoProfile -start FileIOProfile

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start WPR capture" -ForegroundColor Red
    exit 1
}

Write-Host "Capturing performance data for $DurationSeconds seconds..." -ForegroundColor Green

# Run the executable in background
$process = Start-Process -FilePath $ExePath -ArgumentList `
    "-m C:\Users\fabia\.lmstudio\models\lmstudio-community\Qwen3-4B-Instruct-2507-GGUF\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "-f C:\Users\fabia\Projects\kv-compact\prompt-long.txt",
    "-n 50" `
    -PassThru -NoNewWindow

# Wait for process to complete or timeout
$timeouted = $false
try {
    $process.WaitForExit(($DurationSeconds + 10) * 1000)
} catch [System.TimeoutException] {
    $timeouted = $true
    Write-Host "Process timeout, stopping capture..." -ForegroundColor Yellow
    $process.Kill()
}

Write-Host "Process exited with code: $($process.ExitCode)" -ForegroundColor Cyan

# Stop capture
Write-Host "`nStep 2: Stopping capture and saving trace..." -ForegroundColor Yellow
wpr -stop $traceFile

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to stop WPR capture" -ForegroundColor Red
    exit 1
}

Write-Host "`nStep 3: Converting to flamegraph format..." -ForegroundColor Yellow

# Generate flamegraph-compatible CSV
$csvFile = Join-Path $OutputDir "kv-compact_$timestamp.csv"
& "$PSScriptRoot\etw2flamegraph.ps1" -TraceFile $traceFile -OutputFile $csvFile

Write-Host "`n=== Profiling Complete ===" -ForegroundColor Green
Write-Host "Trace file: $traceFile"
Write-Host "CSV file: $csvFile"
Write-Host ""
Write-Host "To view the trace:" -ForegroundColor Yellow
Write-Host "1. Open Windows Performance Analyzer (WPA)"
Write-Host "2. File > Open > $traceFile"
Write-Host "3. Analyze > CPU Usage (Sampled)"
Write-Host ""
Write-Host "To generate a flamegraph:" -ForegroundColor Yellow
Write-Host "1. Use the CSV file with FlameGraph tools"
Write-Host "2. Or view in WPA's Graph view"
