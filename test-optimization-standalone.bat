@echo off
REM Standalone test script for optimization algorithms
REM Compiles and runs the test without requiring CMake

set MSVC=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207
set SDK=C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0
set INCLUDE=%MSVC%\include;%SDK%\ucrt;%SDK%\shared;%SDK%\um;%SDK%\winrt
set LIB=%MSVC%\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.26100.0\um\x64
set PATH=%PATH%;%MSVC%\bin\Hostx64\x64

cd /d C:\Users\fabia\Projects\kv-compact

echo Compiling standalone optimization test...
cl /EHsc /std:c++17 /O2 /Fe:test-optimization-standalone.exe tests/test-optimization-standalone.cpp

if %errorlevel% neq 0 (
    echo COMPILATION FAILED
    exit /b %errorlevel%
)

echo.
echo Running test with 1000 tokens...
test-optimization-standalone.exe --tokens 1000

echo.
echo Running test with 10000 tokens...
test-optimization-standalone.exe --tokens 10000

echo.
echo Running test with 100000 tokens...
test-optimization-standalone.exe --tokens 100000 --quiet
