@echo off
REM ComfyDoctor - double-click me when ComfyUI won't start.
REM
REM This is the whole point of the tool: when torch is broken, ComfyUI never
REM finishes booting, so no node and no sidebar panel can possibly help you.
REM This still runs.
REM
REM Finds the right Python by itself, in the order that is actually correct for
REM a ComfyUI install - the portable embedded interpreter first, because that is
REM the one ComfyUI is really using, and a system `python` on PATH is almost
REM always a different one with a different set of packages.

setlocal
cd /d "%~dp0"
chcp 65001 >nul 2>&1

set "DOCTOR=%~dp0doctor.py"

REM ComfyUI portable: custom_nodes\<us>\ -> up 3 -> python_embeded\
for %%P in (
    "%~dp0..\..\..\python_embeded\python.exe"
    "%~dp0..\..\..\..\python_embeded\python.exe"
    "%~dp0..\..\..\venv\Scripts\python.exe"
    "%~dp0..\..\..\.venv\Scripts\python.exe"
) do (
    if exist "%%~fP" (
        echo Using "%%~fP"
        echo.
        "%%~fP" -s "%DOCTOR%" %*
        goto :done
    )
)

echo Could not find ComfyUI's Python next to this folder.
echo Falling back to whatever "python" is on your PATH - note this may be a
echo DIFFERENT Python than ComfyUI uses, so the results may not reflect what
echo ComfyUI actually sees.
echo.
python "%DOCTOR%" %*

:done
echo.
pause
