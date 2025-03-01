import os
import glob
import numpy as np

# Directory containing the prediction files
input_dir = 'runs/predict/keypoint_detection/labels'
output_dir = 'runs/predict/keypoint_detection/filtered_labels'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Size thresholds for filtering (in relative coordinates)
MIN_WIDTH = 0.03  # Minimum width of bounding box
MAX_WIDTH = 0.08  # Maximum width of bounding box
MIN_HEIGHT = 0.10  # Minimum height of bounding box
MAX_HEIGHT = 0.25  # Maximum height of bounding box

def filter_predictions(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    filtered_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:  # Skip invalid lines
            continue
            
        # Extract bounding box dimensions
        width = float(parts[3])
        height = float(parts[4])
        
        # Apply size thresholds
        if (MIN_WIDTH <= width <= MAX_WIDTH and 
            MIN_HEIGHT <= height <= MAX_HEIGHT):
            filtered_lines.append(line)
    
    # Write filtered predictions
    if filtered_lines:
        with open(output_file, 'w') as f:
            f.writelines(filtered_lines)

# Process all prediction files
for input_file in glob.glob(os.path.join(input_dir, '*.txt')):
    filename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, filename)
    filter_predictions(input_file, output_file)

print("Filtering complete. Filtered predictions saved in:", output_dir) 