@echo off

set PYTHON=python
set POETRY_VERSION=1.8.3


if "%1"=="" goto all
if "%1"=="install" goto install
if "%1"=="run" goto run
if "%1"=="clean" goto clean

:all
call :check_python
call :check_poetry
call :install
call :run
goto :eof

:check_python
%PYTHON% -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>NUL
if errorlevel 1 (
    echo Python 3.11 or higher is required but not found. Please install Python 3.11 or higher.
    exit /b 1
)
echo Python 3.11 or higher is installed.
goto :eof

:check_poetry
poetry --version >NUL 2>&1
if errorlevel 1 (
    echo Poetry not found. Installing Poetry version %POETRY_VERSION%...
    pip install --user poetry==%POETRY_VERSION%
)
goto :eof

:install
echo Creating Poetry environment and installing dependencies...
poetry env use %PYTHON%
poetry install
goto :eof

:run
echo Starting the server...
start /B poetry run python server.py
echo Waiting for server to start...
timeout /t 5 /nobreak >NUL
echo Starting the client...
poetry run python client.py
echo Stopping the server...
taskkill /F /IM python.exe /T
goto :eof

:clean
echo Cleaning up Poetry environment...
poetry env remove --all
echo Removing __pycache__ directories...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Cleanup complete.
goto :eof