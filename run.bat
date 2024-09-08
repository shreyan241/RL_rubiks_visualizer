@echo off

set PYTHON=python
set POETRY_VERSION=1.8.3

if "%1"=="" goto all
if "%1"=="install" goto install
if "%1"=="run" goto run
if "%1"=="runner" goto runner
if "%1"=="clean" goto clean

:all
call :check_python
call :check_poetry
call :install
call :run
goto :eof

:check_python
%PYTHON% --version 2>NUL
if errorlevel 1 (
    echo Python is not installed or not in PATH. Please install Python 3.11.
    exit /b 1
)
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

:runner
echo Starting the runner...
poetry run python runner.py
goto :eof

:clean
echo Cleaning up Poetry environment...
poetry env remove --all
echo Removing __pycache__ directories...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo Cleanup complete.
goto :eof