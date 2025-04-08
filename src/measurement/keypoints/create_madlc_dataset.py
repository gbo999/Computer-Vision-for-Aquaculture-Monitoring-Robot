import os
import pandas as pd
import numpy as np
from pathlib import Path
import deeplabcut as dlc
import cv2
from tqdm import tqdm

def convert_yolo_to_dlc(yolo_label_path, image_path, output_dir, keypoint_names):
    """
    Convert a single YOLO-Pose label to DeepLabCut format.
    
    Args:
        yolo_label_path: Path to the YOLO label file
        image_path: Path to the corresponding image
        output_dir: Directory to save the DLC label
        keypoint_names: List of keypoint names in order
    """
    # Read image dimensions
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = img.shape[:2]
    
    # Read YOLO labels
    with open(yolo_label_path, 'r') as f:
        lines = f.readlines()
    
    # Process each detection
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:  # Skip invalid lines
            continue
            
        # YOLO format: class x_center y_center width height keypoints...
        class_id = int(parts[0])
        x_center, y_center = float(parts[1]), float(parts[2])
        width, height = float(parts[3]), float(parts[4])
        
        # Convert normalized coordinates to pixel coordinates
        x_center = x_center * width
        y_center = y_center * height
        width = width * width
        height = height * height
        
        # Process keypoints (4 keypoints per detection)
        keypoints = []
        likelihoods = []
        for i in range(4):
            idx = 5 + i * 3
            if idx + 2 < len(parts):
                x = float(parts[idx]) * width
                y = float(parts[idx + 1]) * height
                conf = float(parts[idx + 2])
                keypoints.extend([x, y])
                likelihoods.append(conf)
        
        # Create DataFrame for DLC
        df = pd.DataFrame({
            'coords': [keypoints],
            'likelihood': [likelihoods],
            'individuals': ['individual0'],
            'bodyparts': [keypoint_names * 2],  # Each keypoint has x,y
            'scorer': ['manual']
        })
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}.csv")
        df.to_csv(output_path, index=False)
        return output_path

def create_madlc_project(project_name, experimenter, video_path, keypoint_names):
    """
    Create a new maDLC project.
    
    Args:
        project_name: Name of the project
        experimenter: Name of the experimenter
        video_path: Path to a sample video
        keypoint_names: List of keypoint names
    """
    # Create project
    config_path = dlc.create_new_project(
        project_name,
        experimenter,
        [video_path],
        working_directory=os.getcwd(),
        keypoint_names=keypoint_names
    )
    
    # Configure for multi-animal tracking
    cfg = dlc.auxiliaryfunctions.read_config(config_path)
    cfg['multianimalproject'] = True
    cfg['individuals'] = ['individual0']  # Add more individuals as needed
    cfg['skeleton'] = [[0,1], [1,2], [2,3]]  # Define skeleton connections
    dlc.auxiliaryfunctions.write_config(config_path, cfg)
    
    return config_path

def convert_dataset(input_dir, output_dir, keypoint_names):
    """
    Convert an entire dataset of YOLO-Pose labels to DeepLabCut format.
    
    Args:
        input_dir: Directory containing images and labels
        output_dir: Directory to save DLC labels
        keypoint_names: List of keypoint names in order
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(os.path.join(input_dir, 'images')) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Process each image
    for img_file in tqdm(image_files, desc="Converting labels"):
        img_path = os.path.join(input_dir, 'images', img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(input_dir, 'labels', label_file)
        
        if os.path.exists(label_path):
            try:
                convert_yolo_to_dlc(label_path, img_path, output_dir, keypoint_names)
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")

def main():
    # Define keypoint names in order (matching YOLO format)
    keypoint_names = ['start_carapace', 'eyes', 'rostrum', 'tail']
    
    # Create maDLC project
    project_name = "crayfish_tracking"
    experimenter = "researcher"
    sample_video = "test-car/images/GX010183_37_685-jpg_gamma_jpg.rf.d5c334aefde6a9b438421e31ffff19b1.jpg"
    
    config_path = create_madlc_project(project_name, experimenter, sample_video, keypoint_names)
    
    # Convert dataset
    input_dir = "test-car"
    output_dir = "dlc_labels"
    convert_dataset(input_dir, output_dir, keypoint_names)
    
    print(f"Project created at: {config_path}")
    print("Dataset converted successfully!")
    print("\nNext steps:")
    print("1. Check the converted labels in the dlc_labels directory")
    print("2. Use deeplabcut.create_training_dataset() to create the training dataset")
    print("3. Configure training parameters in the config.yaml file")
    print("4. Start training with deeplabcut.train_network()")

if __name__ == "__main__":
    main() 