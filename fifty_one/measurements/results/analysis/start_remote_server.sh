#!/bin/bash

# Define script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default values
TYPE="body"
WEIGHTS_TYPE="all"
ADDRESS="0.0.0.0"
PORT=5151
REMOTE_PORT=5252

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TYPE="$2"
            shift 2
            ;;
        --weights_type)
            WEIGHTS_TYPE="$2"
            shift 2
            ;;
        --address)
            ADDRESS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --remote_port)
            REMOTE_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Get server's IP address for display
if [ "$ADDRESS" == "0.0.0.0" ]; then
    if command -v ifconfig &> /dev/null; then
        SERVER_IP=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -n 1)
    elif command -v ip &> /dev/null; then
        SERVER_IP=$(ip addr | grep 'inet ' | grep -v '127.0.0.1' | awk '{print $2}' | cut -d/ -f1 | head -n 1)
    else
        SERVER_IP="<your-server-ip>"
    fi
    echo "Your server's IP address appears to be: $SERVER_IP"
    echo "Users should connect using: fiftyone app connect --destination $SERVER_IP:$REMOTE_PORT --port $PORT"
    echo ""
fi

# Run the remote server
echo "Starting FiftyOne remote server..."
python3 remote_fiftyone_server.py --type "$TYPE" --weights_type "$WEIGHTS_TYPE" --address "$ADDRESS" --port "$PORT" --remote_port "$REMOTE_PORT" 