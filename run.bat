@echo off

set PYTHONPATH=%cd%

if "%1"=="" goto all
if "%1"=="install" goto install
if "%1"=="run" goto run
if "%1"=="clean" goto clean

:all
call :check_poetry
call :install
call :run
goto :eof

:check_poetry
poetry --version >NUL 2>&1
if errorlevel 1 (
    echo Poetry not found. Please ensure Poetry is installed in the base environment.
    exit /b 1
)
goto :eof

:install
echo Installing dependencies using Poetry...
poetry install
goto :eof

:run
echo Starting the server...
set PYTHONPATH=%cd%
start /B poetry run python server.py
echo Waiting for server to start...
timeout /t 5 /nobreak >NUL
echo Starting the client...
set PYTHONPATH=%cd%
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
