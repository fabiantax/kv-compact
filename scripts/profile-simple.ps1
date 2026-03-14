# Simple profiler using Windows API calls
# Adds instrumentation to measure time spent in each function

Add-Type @"
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

public class Profiler {
    [DllImport("kernel32.dll")]
    public static extern bool QueryPerformanceCounter(out long lpPerformanceCount);

    [DllImport("kernel32.dll")]
    public static extern bool QueryPerformanceFrequency(out long lpFrequency);

    private static long frequency;
    private static long startTime;

    public static void Init() {
        QueryPerformanceFrequency(out frequency);
        QueryPerformanceCounter(out startTime);
    }

    public static double GetElapsedSeconds() {
        long endTime;
        QueryPerformanceCounter(out endTime);
        return (double)(endTime - startTime) / frequency;
    }
}
"@

function Start-Profiling {
    param([string]$Name)

    $script:currentScope = $Name
    $script:scopeStartTime = [Diagnostics::Stopwatch]::StartNew()

    Write-Host "[$Name] Starting..." -ForegroundColor DarkGray
}

function Stop-Profiling {
    param([string]$Name)

    $elapsed = $scopeStartTime.Elapsed.TotalMilliseconds
    Write-Host "[$Name] Completed in ${elapsed}ms" -ForegroundColor DarkGray
}

function Measure-Function {
    param(
        [scriptblock]$ScriptBlock,
        [string]$FunctionName
    )

    $sw = [Diagnostics::Stopwatch]::StartNew()
    & $ScriptBlock
    $sw.Stop()

    Write-Host "[$FunctionName] $($sw.ElapsedMilliseconds)ms" -ForegroundColor Cyan
    return $sw.Elapsed
}

# Export functions
Export-ModuleMember -Function @(
    'Start-Profiling',
    'Stop-Profiling',
    'Measure-Function'
)
