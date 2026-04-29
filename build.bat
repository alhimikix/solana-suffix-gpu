@echo off
setlocal

REM CUDA toolkit. Override via environment if needed.
if "%CUDA_HOME%"=="" set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1

REM MSVC vcvars64.bat. Override via VS_VCVARS env var if installed elsewhere.
if "%VS_VCVARS%"=="" set VS_VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VS_VCVARS%" set VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VS_VCVARS%" set VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VS_VCVARS%" set VS_VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat

if not exist "%CUDA_HOME%\bin\nvcc.exe" (
    echo ERROR: nvcc.exe not found at "%CUDA_HOME%\bin\nvcc.exe"
    echo Set CUDA_HOME environment variable to your CUDA Toolkit path.
    exit /b 1
)

if not exist "%VS_VCVARS%" (
    echo ERROR: vcvars64.bat not found. Install Visual Studio 2019/2022 with C++ workload,
    echo or set VS_VCVARS environment variable to your vcvars64.bat path.
    exit /b 1
)

call "%VS_VCVARS%" >nul

REM -arch=sm_86 targets RTX 30xx (Ampere). For other GPUs:
REM   sm_75 = RTX 20xx, GTX 16xx
REM   sm_86 = RTX 30xx, A100/A40
REM   sm_89 = RTX 40xx, L40
REM   sm_90 = H100, GH200
"%CUDA_HOME%\bin\nvcc.exe" -O3 -arch=sm_86 -allow-unsupported-compiler -Xcompiler "/O2 /MD" -o vanity_gpu.exe vanity.cu
if errorlevel 1 exit /b 1
echo Build OK: vanity_gpu.exe
