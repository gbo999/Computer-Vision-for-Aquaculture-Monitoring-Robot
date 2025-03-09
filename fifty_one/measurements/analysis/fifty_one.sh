#!/bin/bash

# Set default values
TYPE="carapace"
WEIGHTS_TYPE="all"
ERROR_SIZE="mean"

# Define the path to your Python script
PYTHON_SCRIPT="fifty_one/measurements/analysis/measurements_analysis.py"

# Function to display a menu
display_menu() {
    local prompt="$1"
    shift
    local options=("$@")
    
    echo "$prompt"
    for i in "${!options[@]}"; do
        echo "$(($i+1))) ${options[$i]}"
    done
}

# Function to get user input
get_user_input() {
    local num_options="$1"
    local choice
    read -p "Enter your choice [1-$num_options]: " choice
    echo "$choice"
}

# Function to validate and return the selected option
validate_choice() {
    local choice="$1"
    shift
    local options=("$@")
    
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]]; then
        echo "$choice"
    else
        echo "Invalid option. Using default."
        echo "1"
    fi
}

# Function to get choice
get_choice() {
    local prompt="$1"
    shift
    local options=("$@")
    
    display_menu "$prompt" "${options[@]}"
    local choice=$(get_user_input "${#options[@]}")
    validate_choice "$choice" "${options[@]}"
}

# Welcome message
echo "================================"
echo "  Error Flags Analysis Script"
echo "================================"
echo "This script will help you run the flags analysis with your preferred options."
echo

# Get type
TYPE_OPTIONS=("carapace" "body")
display_menu "Select the type:" "${TYPE_OPTIONS[@]}"
TYPE=$(get_user_input "${#TYPE_OPTIONS[@]}")
TYPE=${TYPE_OPTIONS[$((TYPE-1))]}
echo "Selected type: $TYPE"
echo

# Get weights type
WEIGHTS_OPTIONS=("all" "kalkar" "car")
display_menu "Select the weights type:" "${WEIGHTS_OPTIONS[@]}"
WEIGHTS_TYPE=$(get_user_input "${#WEIGHTS_OPTIONS[@]}")
WEIGHTS_TYPE=${WEIGHTS_OPTIONS[$((WEIGHTS_TYPE-1))]}
echo "Selected weights type: $WEIGHTS_TYPE"
echo



# Build the command
CMD="python $PYTHON_SCRIPT --type $TYPE --weights_type $WEIGHTS_TYPE --error_size $ERROR_SIZE"

# Show the command to be executed
echo -e "\nThe following command will be executed:"
echo "$CMD"
echo

# Ask for confirmation
read -p "Do you want to proceed? (y/n): " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Running analysis..."
    # Execute the command
    eval "$CMD"
else
    echo "Operation cancelled."
fi
