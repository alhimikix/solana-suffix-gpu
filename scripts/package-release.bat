@echo off
REM Build all SM architectures and package into release zip.
REM Usage: scripts\package-release.bat v1.0.0
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: %~nx0 ^<version^>
    echo Example: %~nx0 v1.0.0
    exit /b 1
)

set VERSION=%~1
set OUTDIR=release\solana-vanity-gpu-%VERSION%-windows-x64

if "%CUDA_HOME%"=="" set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
if "%VS_VCVARS%"=="" set VS_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VS_VCVARS%" set VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat

if not exist "%CUDA_HOME%\bin\nvcc.exe" (
    echo ERROR: nvcc.exe not found at "%CUDA_HOME%\bin\nvcc.exe"
    exit /b 1
)
if not exist "%VS_VCVARS%" (
    echo ERROR: vcvars64.bat not found
    exit /b 1
)

call "%VS_VCVARS%" >nul
mkdir "%OUTDIR%" 2>nul

for %%A in (75 86 89) do (
    echo Building sm_%%A...
    "%CUDA_HOME%\bin\nvcc.exe" -O3 -arch=sm_%%A -allow-unsupported-compiler -Xcompiler "/O2 /MD" -o "%OUTDIR%\vanity_gpu_sm%%A.exe" vanity.cu
    if errorlevel 1 (
        echo Build failed for sm_%%A
        exit /b 1
    )
)

copy README.md "%OUTDIR%\" >nul
copy LICENSE "%OUTDIR%\" >nul
copy CHANGELOG.md "%OUTDIR%\" >nul
copy build.bat "%OUTDIR%\" >nul

cd release
powershell -NoProfile -Command "Compress-Archive -Force -Path 'solana-vanity-gpu-%VERSION%-windows-x64\*' -DestinationPath 'solana-vanity-gpu-%VERSION%-windows-x64.zip'"
cd ..

echo.
echo Done: release\solana-vanity-gpu-%VERSION%-windows-x64.zip
