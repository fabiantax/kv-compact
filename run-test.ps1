# Set up environment
$MSVC = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
$SDK = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0"

$env:INCLUDE = "$MSVC\include;$SDK\ucrt;$SDK\shared;$SDK\um"
$env:LIB = "$MSVC\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64"
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;$MSVC\bin\Hostx64\x64;" + $env:PATH

# Go to project directory
Set-Location C:\Users\fabia\Projects\kv-compact

Write-Host "=== Compiling standalone optimization test ===" -ForegroundColor Cyan
& cl.exe /EHsc /std:c++17 /O2 /Fe:test-opt.exe tests\test-optimization-standalone.cpp 2>&1

if (Test-Path test-opt.exe) {
    Write-Host ""
    Write-Host "=== Test 1: 1000 tokens ===" -ForegroundColor Green
    .\test-opt.exe --tokens 1000

    Write-Host ""
    Write-Host "=== Test 2: 10000 tokens ===" -ForegroundColor Green
    .\test-opt.exe --tokens 10000 --quiet

    Write-Host ""
    Write-Host "=== Test 3: 100000 tokens ===" -ForegroundColor Green
    .\test-opt.exe --tokens 100000 --quiet
} else {
    Write-Host "Compilation failed - test-opt.exe not found" -ForegroundColor Red
}
