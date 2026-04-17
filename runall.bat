@echo off

echo Starting server...
start cmd /B /k poetry run python server.py

echo Waiting for server to start...
:check_server
ping -n 1 localhost >nul
curl -sS http://localhost:5000 >nul
if errorlevel 1 (
    goto check_server
)

echo Server started. Starting client...
start cmd /B /k poetry run python client.py
start cmd /B /k poetry run python client.py
start cmd /B /k poetry run python client.py
start cmd /B /k poetry run python client.py
start cmd /B /k poetry run python client.py
echo Client started. Exiting batch script.