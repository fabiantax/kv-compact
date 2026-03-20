@echo off
REM Benchmark Phi-3.5-mini to validate pure attention speedup hypothesis
REM Phi-3.5 is 3.8B, pure attention transformer (NOT hybrid)

setlocal

set MODEL=..\models\Phi-3.5-mini-instruct-Q4_K_M.gguf
set BINARY=build-full\Release\llama-hip-bin\llama-cli.exe

if not exist "%MODEL%" (
    echo ERROR: Model not found at %MODEL%
    echo Please download first:
    echo wget https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf
    exit /b 1
)

echo.
echo ========================================
echo Phi-3.5-mini Benchmark (Pure Attention)
echo ========================================
echo.
echo This test validates that kv-compact works on pure attention models
echo Expected: 2-5x speedup vs hybrid models
echo.

%BINARY% ^
    -m "%MODEL%" ^
    -c 2048 -n 256 ^
    -p "Explain quantum entanglement in simple terms." ^
    -ngl 99 -t 8 ^
    --perf

echo.
echo ========================================
echo Benchmark Complete
echo ========================================
