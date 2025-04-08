#!/bin/bash

# FiftyOne app port - change if needed
PORT=5188
PROXY_PORT=8080

# Check if socat is installed
if ! command -v socat &> /dev/null; then
    echo "socat is not installed. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install socat
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update && sudo apt-get install -y socat
    else
        echo "Could not determine OS type. Please install socat manually."
        exit 1
    fi
fi

# Generate a random subdomain suffix
RANDOM_SUFFIX=$(cat /dev/urandom | LC_ALL=C tr -dc 'a-z0-9' | fold -w 6 | head -n 1)
SUBDOMAIN="fiftyone-app-${RANDOM_SUFFIX}"

# Start socat to proxy from 0.0.0.0:PROXY_PORT to localhost:PORT
echo "Starting socat proxy from 0.0.0.0:$PROXY_PORT to localhost:$PORT..."
socat TCP-LISTEN:$PROXY_PORT,fork TCP:localhost:$PORT &
SOCAT_PID=$!

# Make sure to kill socat when script exits
trap "kill $SOCAT_PID 2>/dev/null; echo 'Proxy stopped.'; exit" INT TERM EXIT

echo "Starting localtunnel for proxy on port $PROXY_PORT..."
echo "Using subdomain: $SUBDOMAIN"

# Run localtunnel for the proxy port
npx localtunnel --port $PROXY_PORT --subdomain $SUBDOMAIN

# This code only runs if localtunnel exits
echo "Tunnel closed."
kill $SOCAT_PID 2>/dev/null
echo "Proxy stopped." 