#!/bin/bash

PYTHONPATH=$(pwd)

install() {
    echo "Installing dependencies using Poetry..."
    poetry install
}

run() {
    echo "Starting the server..."
    poetry run python server.py &
    SERVER_PID=$!
    echo "Waiting for server to start..."
    sleep 5
    echo "Starting the client..."
    poetry run python client.py
    echo "Stopping the server..."
    kill $SERVER_PID
}

clean() {
    echo "Cleaning up Poetry environment..."
    poetry env remove --all
    echo "Removing __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    echo "Cleanup complete."
}

case "$1" in
    install)
        install
        ;;
    run)
        run
        ;;
    clean)
        clean
        ;;
    *)
        install
        run
        ;;
esac
