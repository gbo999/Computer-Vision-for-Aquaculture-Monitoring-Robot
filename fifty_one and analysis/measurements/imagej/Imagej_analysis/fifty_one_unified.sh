#!/bin/bash

echo "================================"
echo "  Unified Analysis Script"
echo "================================"
echo "This script will help you run the unified analysis with your preferred options."

# Select weights type
echo -e "\nSelect the weights type:"
echo "1) all"
echo "2) kalkar"
echo "3) car"
read -p "Enter your choice [1-3]: " weights_choice

case $weights_choice in
    1)
        weights_type="all"
        ;;
    2)
        weights_type="kalkar"
        ;;
    3)
        weights_type="car"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "Selected weights type: $weights_type"

# Get a random port between 5150 and 5190
port=$(( ( RANDOM % 41 ) + 5150 ))
echo "Using port: $port"

# Construct and display the command
command="python3 \"$(pwd)/measurements_analysis_unified.py\" --weights_type $weights_type --port $port"
echo -e "\nExecuting command:\n$command\n"

# Execute the command
eval $command 