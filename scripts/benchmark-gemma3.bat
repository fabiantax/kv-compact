@echo off
REM Benchmark Gemma-3-4B (Pure Attention) to validate kv-compact speedup
REM Gemma 3 is Google's latest pure transformer (NOT hybrid)

setlocal

set MODEL=..\models\gemma-3-4b-it-heretic-v1.2-Q4_K_M.gguf
set BINARY=build-full\Release\llama-hip-bin\llama-cli.exe

if not exist "%MODEL%" (
    echo ERROR: Model not found at %MODEL%
    echo Download from: https://huggingface.co/grayarea/gemma-3-4b-it-heretic-v1.2-GGUF
    exit /b 1
)

echo.
echo ========================================
echo Gemma-3-4B Benchmark (Pure Attention)
echo ========================================
echo.
echo This test validates kv-compact on 100%% attention layers
echo Expected: 2-5x speedup vs hybrid Qwen 3.5
echo.

%BINARY% ^
    -m "%MODEL%" ^
    -c 2048 -n 256 ^
    -p "Explain quantum computing in simple terms." ^
    -ngl 99 -t 8 ^
    --perf

echo.
echo ========================================
echo Comparison Notes
echo ========================================
echo Qwen 3.5 (75%% DeltaNet, 25%% attention): ~24 t/s
echo Gemma 3 (100%% attention): Expected 50-100 t/s with kv-compact
echo.
