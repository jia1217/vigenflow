@echo off
setlocal

cd /d "%~dp0"

set "OUT_DIR=vigenflow_win"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

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
if defined BOOST_FILESYSTEM_LIB (
    set "REQUESTED_BOOST_FILESYSTEM_LIB=%BOOST_FILESYSTEM_LIB%"
) else (
    set "REQUESTED_BOOST_FILESYSTEM_LIB=boost_filesystem-vc143-mt-x64-1_88.lib"
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
set "BOOST_FILESYSTEM_LIB=%REQUESTED_BOOST_FILESYSTEM_LIB%"

if exist "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\include" goto have_vcpkg_include
echo VCPKG include path not found: "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\include"
echo Set VCPKG_ROOT or VCPKG_TRIPLET before running this script.
exit /b 1

:have_vcpkg_include
if exist "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\lib\%BOOST_FILESYSTEM_LIB%" goto have_boost_filesystem
for %%F in ("%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\lib\boost_filesystem-*.lib") do set "BOOST_FILESYSTEM_LIB=%%~nxF"
if exist "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\lib\%BOOST_FILESYSTEM_LIB%" goto have_boost_filesystem
echo Boost filesystem library not found: "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\lib\%BOOST_FILESYSTEM_LIB%"
echo Set BOOST_FILESYSTEM_LIB before running this script if your Boost version is different.
exit /b 1

:have_boost_filesystem

cl /utf-8 /O2 /EHsc /MD /std:c++17 /D_WIN32_WINNT=0x0A00 /DWIN32_LEAN_AND_MEAN ^
  /I . /I .\include /I "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\include" ^
  server_main.cpp server_config.cpp file_utils.cpp request_parser.cpp z_image_bf16_export.cpp flux_klein_bf16_export.cpp worker.cpp http_handlers.cpp tcp_server.cpp cli.cpp ^
  /Fo"%OUT_DIR%\\" /Fe"%OUT_DIR%\vgf-serve.exe" ^
  /link "/LIBPATH:%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\lib" ws2_32.lib "%BOOST_FILESYSTEM_LIB%"
if errorlevel 1 exit /b 1

set "BOOST_FILESYSTEM_DLL=%BOOST_FILESYSTEM_LIB:.lib=.dll%"
set "BOOST_SYSTEM_DLL=%BOOST_FILESYSTEM_DLL:boost_filesystem=boost_system%"
if exist "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\bin\%BOOST_FILESYSTEM_DLL%" copy /Y "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\bin\%BOOST_FILESYSTEM_DLL%" "%OUT_DIR%\" >nul
if exist "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\bin\%BOOST_SYSTEM_DLL%" copy /Y "%VCPKG_ROOT%\installed\%VCPKG_TRIPLET%\bin\%BOOST_SYSTEM_DLL%" "%OUT_DIR%\" >nul

echo Built %CD%\%OUT_DIR%\vgf-serve.exe
