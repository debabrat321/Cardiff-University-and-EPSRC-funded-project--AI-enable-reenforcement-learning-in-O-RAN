#!/bin/bash

echo "Starting server..."
poetry run python server.py &

echo "Waiting for server to start..."
while ! curl -sS "http://localhost:5000" > /dev/null; do
    sleep 1
done

echo "Server started. Starting clients..."
for i in {1..5}; do
    gnome-terminal -- bash -c "poetry run python client.py; echo 'Press [ENTER] to close this terminal window'; read"
done

echo "Clients started. Exiting shell script."
