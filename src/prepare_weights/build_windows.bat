@echo off
rem Builds vigenflow_win\prepare_base.exe with MSVC (nlohmann-json from vcpkg; curl.exe at run time, part of
rem Windows 10 1803+). Stages the current model catalog beside the standalone executable.
rem Pass --test to run the local download/unpack regression suite after building (requires Python).
setlocal

cd /d "%~dp0"

set "PREPARE_RUN_TESTS=0"
if "%~1"=="" goto options_done
if /i not "%~1"=="--test" goto usage
if not "%~2"=="" goto usage
set "PREPARE_RUN_TESTS=1"
:options_done

set "OUT_DIR=vigenflow_win"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist build-windows mkdir build-windows

if defined VCPKG_ROOT (
    set "REQUESTED_VCPKG_ROOT=%VCPKG_ROOT%"
) else (
    set "REQUESTED_VCPKG_ROOT=C:\dev\vcpkg"
)
if defined VCPKG_TRIPLET (
    set "REQUESTED_VCPKG_TRIPLET=%VCPKG_TRIPLET%"
) else (
    set "REQUESTED_VCPKG_TRIPLET=x64-windows"
)
set "PREPARE_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

where cl >nul 2>nul
if not errorlevel 1 if /i "%VSCMD_ARG_TGT_ARCH%"=="x64" goto have_cl

if exist "%PREPARE_VSWHERE%" goto find_visual_studio
echo Could not find cl.exe or vswhere.exe. Open a Developer Command Prompt or install Visual Studio C++ tools.
exit /b 1

:find_visual_studio
set "PREPARE_VSINSTALL="
for /f "usebackq tokens=*" %%I in (`"%PREPARE_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "PREPARE_VSINSTALL=%%I"
if defined PREPARE_VSINSTALL goto have_visual_studio
echo Could not find Visual Studio C++ tools.
exit /b 1

:have_visual_studio
if exist "%PREPARE_VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" goto have_vcvars
echo Could not find "%PREPARE_VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat".
exit /b 1

:have_vcvars
call "%PREPARE_VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

:have_cl
set "VCPKG_ROOT=%REQUESTED_VCPKG_ROOT%"
set "VCPKG_TRIPLET=%REQUESTED_VCPKG_TRIPLET%"

if exist "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\include\nlohmann\json.hpp" goto have_json
echo nlohmann/json.hpp not found under "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\include"
echo Install it with vcpkg install nlohmann-json:%VCPKG_TRIPLET%, or set VCPKG_ROOT / VCPKG_TRIPLET.
exit /b 1

:have_json
cl /nologo /utf-8 /O2 /EHsc /MT /std:c++20 /permissive- /W3 /D_CRT_SECURE_NO_WARNINGS ^
  /I "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\include" ^
  src\prepare_base_main.cpp src\packed_format.cpp src\hf_download.cpp ^
  /Fo"build-windows\\" /Fe"%OUT_DIR%\prepare_base.exe" ^
  /link /MANIFEST:EMBED /MANIFESTINPUT:src\prepare.manifest
if errorlevel 1 exit /b 1

copy /y prepare_models.jsonc "%OUT_DIR%\prepare_models.jsonc" >nul
if errorlevel 1 exit /b 1
echo Built "%CD%\%OUT_DIR%\prepare_base.exe"
echo Model catalog: "%CD%\%OUT_DIR%\prepare_models.jsonc"
if "%PREPARE_RUN_TESTS%"=="0" exit /b 0

rem The Python launcher can exist without an installed interpreter; probe it before choosing it.
py -3 -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>nul
if not errorlevel 1 goto test_with_py
python -c "import sys; sys.exit(sys.version_info < (3, 9))" >nul 2>nul
if not errorlevel 1 goto test_with_python
echo Python 3.9 or newer is required for --test. Install it or build without --test.
exit /b 1

:test_with_py
py -3 tests\test_prepare_base.py "%OUT_DIR%\prepare_base.exe"
exit /b %errorlevel%

:test_with_python
python tests\test_prepare_base.py "%OUT_DIR%\prepare_base.exe"
exit /b %errorlevel%

:usage
echo Usage: build_windows.bat [--test]
exit /b 2
