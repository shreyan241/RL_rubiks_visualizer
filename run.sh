#!/bin/bash

REQUIRED_PYTHON_VERSION="3.11"
POETRY_VERSION="1.8.3"

check_python_version() {
    echo "Checking Python version..."
    if command -v python3 &>/dev/null; then
        PYTHON_CMD=python3
    elif command -v python &>/dev/null; then
        PYTHON_CMD=python
    else
        echo "Python is not installed. Please install Python $REQUIRED_PYTHON_VERSION."
        exit 1
    fi

    INSTALLED_PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    if [ "$(printf '%s\n' "$REQUIRED_PYTHON_VERSION" "$INSTALLED_PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_PYTHON_VERSION" ]; then
        echo "Python $REQUIRED_PYTHON_VERSION or higher is required, but $INSTALLED_PYTHON_VERSION is installed."
        exit 1
    fi
    echo "Python version check passed. Using Python $INSTALLED_PYTHON_VERSION."
}

install() {
    echo "Installing dependencies..."
    $PYTHON_CMD -m pip install "poetry==$POETRY_VERSION"
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
    echo "Removing __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    echo "Cleanup complete."
}

check_python_version

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