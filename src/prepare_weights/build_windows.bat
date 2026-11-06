@echo off
rem Builds vigenflow_win\prepare_base.exe with MSVC (nlohmann-json from vcpkg; curl.exe at run time, part of
rem Windows 10 1803+). The static C runtime and the embedded long-path manifest make it a single file to copy.
setlocal

cd /d "%~dp0"

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
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

where cl >nul 2>nul
if not errorlevel 1 goto have_cl

if exist "%VSWHERE%" goto find_visual_studio
echo Could not find cl.exe or vswhere.exe. Open a Developer Command Prompt or install Visual Studio C++ tools.
exit /b 1

:find_visual_studio
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
if defined VSINSTALL goto have_visual_studio
echo Could not find Visual Studio C++ tools.
exit /b 1

:have_visual_studio
if exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" goto have_vcvars
echo Could not find "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat".
exit /b 1

:have_vcvars
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
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

echo Built %CD%\%OUT_DIR%\prepare_base.exe
