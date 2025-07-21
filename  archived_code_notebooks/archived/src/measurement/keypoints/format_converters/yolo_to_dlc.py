import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def convert_yolo_to_dlc(yolo_label_path, image_path, output_dir, keypoint_names):
    """
    Convert YOLO-Pose format labels to DeepLabCut format.
    
    Args:
        yolo_label_path (str): Path to YOLO format label file
        image_path (str): Path to corresponding image file
        output_dir (str): Directory to save DeepLabCut format labels
        keypoint_names (list): List of keypoint names in order
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read image dimensions
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_height, img_width = img.shape[:2]
    
    # Read YOLO labels
    if not os.path.exists(yolo_label_path):
        print(f"Warning: Label file not found: {yolo_label_path}")
        return
        
    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()
    
    # Initialize lists to store data
    data = []
    
    # Process each line (each line represents one object with keypoints)
    for line in lines:
        values = line.strip().split()
        if len(values) < 5:  # Skip invalid lines
            continue
            
        # Parse class and bbox (first 5 values)
        class_id = int(values[0])
        x_center, y_center, width, height = map(float, values[1:5])
        
        # Convert normalized coordinates to pixel coordinates
        x_center = x_center * img_width
        y_center = y_center * img_height
        width = width * img_width
        height = height * img_height
        
        # Parse keypoints (remaining values)
        keypoints = values[5:]
        if len(keypoints) % 3 != 0:
            print(f"Warning: Invalid keypoint format in {yolo_label_path}")
            continue
            
        # Process keypoints
        keypoint_data = {}
        for i in range(0, len(keypoints), 3):
            kp_idx = i // 3
            if kp_idx >= len(keypoint_names):
                break
                
            x = float(keypoints[i]) * img_width
            y = float(keypoints[i + 1]) * img_height
            conf = float(keypoints[i + 2])
            
            # Only include keypoints with confidence > 0
            if conf > 0:
                keypoint_data[f"{keypoint_names[kp_idx]}_x"] = x
                keypoint_data[f"{keypoint_names[kp_idx]}_y"] = y
                keypoint_data[f"{keypoint_names[kp_idx]}_likelihood"] = conf
        
        if keypoint_data:
            data.append(keypoint_data)
    
    if data:
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add required DeepLabCut columns
        df.insert(0, 'scorer', 'human')
        df.insert(1, 'bodyparts', 'single')
        df.insert(2, 'coords', 'x')
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"{Path(image_path).stem}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved DeepLabCut format labels to: {output_path}")

def convert_dataset(input_dir, output_dir, keypoint_names):
    """
    Convert an entire dataset of YOLO-Pose labels to DeepLabCut format.
    
    Args:
        input_dir (str): Directory containing images and YOLO labels
        output_dir (str): Directory to save DeepLabCut format labels
        keypoint_names (list): List of keypoint names in order
    """
    # Find all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(input_dir).glob(f'**/*{ext}'))
    
    for image_path in tqdm(image_files, desc="Converting labels"):
        # Construct corresponding label path
        label_path = str(image_path).replace(ext, '.txt')
        
        # Convert labels
        convert_yolo_to_dlc(
            label_path,
            str(image_path),
            output_dir,
            keypoint_names
        )

if __name__ == "__main__":
    # Example usage
    input_dir = "path/to/input/directory"  # Directory containing images and YOLO labels
    output_dir = "path/to/output/directory"  # Directory to save DeepLabCut format labels
    
    # Define keypoint names in order (matching YOLO format)
    keypoint_names = [
        "start_carapace",
        "eyes",
        "rostrum",
        "tail"
    ]
    
    convert_dataset(input_dir, output_dir, keypoint_names) 