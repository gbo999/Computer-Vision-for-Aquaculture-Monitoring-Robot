import os
import pandas as pd
import numpy as np
import fiftyone as fo
import fiftyone.utils.data as foud
from fiftyone import ViewField as F
import glob
from PIL import Image  # Add PIL for image operations

# Use absolute paths for everything to avoid path issues
BASE_DIR = "/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms"
LABELS_DIR = os.path.join(BASE_DIR, "runs/pose/predict80/labels")
IMAGES_DIR = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized"
CSV_FILE = os.path.join(BASE_DIR, "runs/pose/predict80/length_analysis_new.csv")

# Create a new dataset
dataset_name = "prawn_keypoints"
dataset = fo.Dataset(dataset_name, overwrite=True)

# The keypoint skeleton definition - use the one from YOLO format
dataset.default_skeleton = fo.KeypointSkeleton(
    labels=["start_carapace", "eyes", "rostrum", "tail"],
    edges=[
        [0, 1],  # start_carapace to eyes
        [1, 2],  # eyes to rostrum
        [0, 3]   # start_carapace to tail
    ]
)

def process_poses(poses, is_ground_truth=False):
    """
    Process YOLO keypoint detections into FiftyOne format.

    Args:
        poses (list): List of YOLO pose detections
            Format: [class_id, x, y, w, h, kp1_x, kp1_y, kp1_conf, kp2_x, kp2_y, kp2_conf, ...]
        is_ground_truth (bool): Whether these are ground truth annotations

    Returns:
        tuple: (keypoints_list, detections)
            - keypoints_list: List of FiftyOne Keypoint objects
            - detections: List of FiftyOne Detection objects

    Keypoint order in YOLO format:
        0: start_carapace
        1: eyes
        2: rostrum
        3: tail
    """
    keypoints_list = []
    detections = []

    for pose in poses:
        if len(pose) == 17:  # 1 class + 4 bbox + 4 keypoints × 3 values
            # Bounding box processing
            x1_rel, y1_rel, width_rel, height_rel = pose[1:5]
            x1_rel -= width_rel / 2
            y1_rel -= height_rel / 2

            # Process keypoints
            keypoints = [[pose[i], pose[i + 1]] for i in range(5, len(pose), 3)]
            keypoint = fo.Keypoint(points=keypoints)
            keypoints_list.append(keypoint)

            # Create keypoints dictionary with correct order
            keypoints_dict = {
                'start_carapace': keypoints[0],  # Index 0 in YOLO format
                'eyes': keypoints[1],            # Index 1 in YOLO format
                'rostrum': keypoints[2],         # Index 2 in YOLO format
                'tail': keypoints[3],            # Index 3 in YOLO format
                'keypoint_ID': keypoint.id
            }

            # Create detection
            if not is_ground_truth:
                detections.append(fo.Detection(
                    label="prawn",
                    bounding_box=[x1_rel, y1_rel, width_rel, height_rel],
                    attributes={'keypoints': keypoints_dict}
                ))
            else:
                detections.append(fo.Detection(
                    label=f"prawn_truth",
                    bounding_box=[x1_rel, y1_rel, width_rel, height_rel],
                    attributes={'keypoints': keypoints_dict}
                ))
    
    return keypoints_list, detections

# Function to parse YOLO keypoints file and process using process_poses
def parse_and_process_keypoints(filepath, img_width, img_height):
    """
    Parse a YOLO keypoints file and process the poses
    
    Args:
        filepath (str): Path to the YOLO keypoints file
        img_width (int): Image width for scaling normalized coordinates
        img_height (int): Image height for scaling normalized coordinates
        
    Returns:
        tuple: Processed keypoints list and detections
    """
    if not os.path.exists(filepath):
        return [], []
        
    poses = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        pose = [float(x) for x in line.strip().split()]
        # Scale the normalized coordinates to image dimensions
        scaled_pose = []
        for i, val in enumerate(pose):
            if i == 0:  # class ID
                scaled_pose.append(val)
            elif i % 3 == 1 or i % 3 == 2:  # x, y coordinates
                if i % 3 == 1:  # x coordinate
                    scaled_pose.append(val)
                else:  # y coordinate
                    scaled_pose.append(val)
            else:  # confidence values
                scaled_pose.append(val)
        poses.append(scaled_pose)
    
    return process_poses(poses)

# Load the CSV file
df = pd.read_csv(CSV_FILE)

# Process each image
processed_count = 0
for index, row in df.iterrows():
    image_name = row['image_name']
    
    # Skip rows with no valid measurement data
    if (pd.isna(row['big_total_length']) and pd.isna(row['small_total_length'])):
        continue
    
    # Construct paths
    image_path = os.path.join(IMAGES_DIR, f"{image_name.replace('colored_', '')}.jpg")
    label_path = os.path.join(LABELS_DIR, f"{image_name}.txt")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
    
    # Create FiftyOne sample
    sample = fo.Sample(filepath=image_path)
    
    # Get image dimensions using PIL
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Error reading image dimensions for {image_path}: {e}")
        img_width = 1920  # Default fallback width
        img_height = 1080  # Default fallback height
    
    # Add metadata from CSV
    sample["metadata"] = fo.ImageMetadata(width=img_width, height=img_height)
    sample["tags"] = []
    
    # Add measurements to sample from CSV
    pond_type = "Circle" if '10191' in image_name else "Square"
    sample["tags"].append(pond_type)
    
    # Process keypoints if label file exists
    if os.path.exists(label_path):
        # Use the process_poses function
        keypoints_list, detections = parse_and_process_keypoints(label_path, img_width, img_height)
        
        # Add CSV data to detections for measurement information
        if len(detections) > 0:
            # Add measurement data to detections
            for detection in detections:
                if not pd.isna(row['big_total_length']):
                    detection["total_length"] = float(row['big_total_length'])
                    detection["carapace_length"] = float(row['big_carapace_length']) if not pd.isna(row['big_carapace_length']) else None
                    sample["tags"].append(f"big_prawn_{row['big_total_length']:.1f}mm")
                    detection.label = "big_prawn"
                elif not pd.isna(row['small_total_length']):
                    detection["total_length"] = float(row['small_total_length'])
                    detection["carapace_length"] = float(row['small_carapace_length']) if not pd.isna(row['small_carapace_length']) else None
                    sample["tags"].append(f"small_prawn_{row['small_total_length']:.1f}mm")
                    detection.label = "small_prawn"
            
            # Add keypoints and detections to sample
            sample["keypoints"] = fo.Keypoints(keypoints=keypoints_list)
            sample["detections"] = fo.Detections(detections=detections)
    
    # Set a meaningful name for display in the app
    sample["name"] = f"{os.path.splitext(os.path.basename(image_path))[0]} - {pond_type}"
    
    # Add sample to dataset
    dataset.add_sample(sample)
    processed_count += 1
    
    if processed_count % 10 == 0:
        print(f"Processed {processed_count} images")

print(f"Dataset created with {len(dataset)} samples")
print("You can now view the dataset with: dataset.app()")

# Create some views for analysis
try:
    # Create views based on pond type
    circle_pond_view = dataset.match_tags("Circle")
    square_pond_view = dataset.match_tags("Square")
    
    print(f"Found {len(circle_pond_view)} samples from Circle pond")
    print(f"Found {len(square_pond_view)} samples from Square pond")
except Exception as e:
    print(f"Error creating views: {str(e)}")

# Save the dataset
dataset.persistent = True
dataset.save() 

# After creating the dataset, add a default view to show keypoints
print("Setting up default view options for keypoints display...")
session = fo.launch_app(dataset, port=5156)

# Configure the view to show keypoints properly
session.view = dataset.view()
session.config.show_confidence = True
session.config.show_attributes = True
session.config.show_keypoints = True
session.refresh()

print("View configured to show keypoints. Access the app in your browser.")
session.wait()