@echo off
REM Flashcat Betting - Windows launcher. Double-click to run.

setlocal
cd /d "%~dp0"

echo.
echo ========================================================
echo   FLASHCAT BETTING - Windows launcher
echo ========================================================
echo.

REM --- Check Python --------------------------------------------------------
set PYTHON=
for %%P in (python py python3) do (
  %%P -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" 2>NUL
  if not errorlevel 1 (
    set PYTHON=%%P
    goto :found
  )
)

echo Could not find Python 3.11 or newer.
echo Install from https://www.python.org/downloads/ and run again.
pause
exit /b 1

:found
echo Using Python: %PYTHON%

if not exist .venv (
  echo.
  echo Creating virtual environment...
  %PYTHON% -m venv .venv
)
call .venv\Scripts\activate.bat

echo.
echo Installing dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo.
echo Running backtest, reweight, and site build...
set PYTHONPATH=src
python -m flashcat all --start 2023-09-01 --end 2024-02-15 --sport nfl --days-ahead 2

echo.
echo  Done. Open docs\index.html in your browser.
pause
