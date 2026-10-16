@echo off
rem Builds the Windows model launchers ..\exe_models_win\<model>\run.exe from <model>\host.cpp, the same
rem launcher source as the Ubuntu run.exe. Each one calls <model>_host_main in the prebuilt
rem ..\lib\host_<model>.dll (linked through ..\lib\host_<model>.lib), which windows_launcher.cpp loads
rem from ..\..\lib\ beside the launcher's folder. The static C runtime lets run.exe start without DLLs
rem beside it, and windows_launcher.manifest gives it the UTF-8 code page.
rem   build_windows.bat                     all five models
rem   build_windows.bat flux2_klein_4B ...  only these
setlocal

cd /d "%~dp0"

set "MODELS=flux1_schnell flux2_klein_4B flux2_klein_4B_edit krea2_turbo z_image_turbo"
if not "%~1"=="" set "MODELS=%*"
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
for %%M in (%MODELS%) do (
    if not exist "%%M\host.cpp" (
        echo Unknown model %%M: no %%M\host.cpp
        exit /b 1
    )
    if not exist "..\lib\host_%%M.lib" (
        echo Missing ..\lib\host_%%M.lib. Copy host_%%M.dll and host_%%M.lib into src\lib before building.
        exit /b 1
    )
    if not exist "build-windows\%%M" mkdir "build-windows\%%M"
    if not exist "..\exe_models_win\%%M" mkdir "..\exe_models_win\%%M"
    cl /nologo /utf-8 /O2 /EHsc /MT /std:c++20 /W3 %%M\host.cpp windows_launcher.cpp ^
      /Fo"build-windows\%%M\\" /Fe"..\exe_models_win\%%M\run.exe" ^
      /link "..\lib\host_%%M.lib" delayimp.lib /DELAYLOAD:host_%%M.dll ^
      /MANIFEST:EMBED /MANIFESTINPUT:windows_launcher.manifest
    if errorlevel 1 exit /b 1
    echo Built %CD%\..\exe_models_win\%%M\run.exe
)
