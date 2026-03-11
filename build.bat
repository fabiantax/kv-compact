@echo off
setlocal
set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
set INCLUDE=%MSVC%\include;%SDK%\ucrt;%SDK%\shared;%SDK%\um;%SDK%\winrt
set LIB=%MSVC%\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64
set PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%MSVC%\bin\Hostx64\x64
set BUILD_DIR=C:\Users\fabia\Projects\kv-compact\build-full
set LOG=C:\Users\fabia\Projects\kv-compact\build.log
del "%LOG%" 2>nul
if not exist "%BUILD_DIR%\build.ninja" (
    echo === Configuring === > "%LOG%" 2>&1
    cmake -G Ninja -S C:\Users\fabia\Projects\kv-compact -B "%BUILD_DIR%" -DLLAMA_CPP_DIR=C:\Users\fabia\Projects\llama.cpp\llama-flash-attn -DCMAKE_BUILD_TYPE=Release >> "%LOG%" 2>&1
    if errorlevel 1 (
        echo CONFIG FAILED >> "%LOG%"
        exit /b 1
    )
)
echo === Building === >> "%LOG%" 2>&1
cmake --build "%BUILD_DIR%" --target llama-kv-compact test-kv-compact-e2e >> "%LOG%" 2>&1
if errorlevel 1 (
    echo BUILD FAILED >> "%LOG%"
    exit /b 1
)
echo === Build OK === >> "%LOG%"
