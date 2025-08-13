"""
This script processes prawn keypoint data using the FiftyOne library. It loads image and label data, processes keypoints
from YOLO format into FiftyOne format, and creates a dataset for visualization and analysis. The script handles both 
predicted and ground truth keypoints, calculates distances between keypoints, and matches them with ground truth data 
from a CSV file. It also exports the dataset for further use and provides options to view the dataset in a web app. 
The script is designed to work with specific directory structures and file naming conventions, and it includes error 
handling for missing files and data inconsistencies.
"""

import os
import random
import pandas as pd
import numpy as np
import fiftyone as fo
import fiftyone.utils.data as foud
from fiftyone import ViewField as F
import glob
from PIL import Image  # Add PIL for image operations

# Use absolute paths for everything to avoid path issues
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
LABELS_DIR = os.path.join(BASE_DIR, "training and val output/runs/pose/predict83/labels")
# IMAGES_DIR = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized"
# Keep OneDrive path for data sharing
IMAGES_DIR = "OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized"
CSV_FILE = os.path.join(BASE_DIR, "fifty_one_and_analysis/measurements/exuviae/spreadsheet_files/length_analysis_new_split.csv")
EXPORTED_DATASET_DIR = os.path.join(BASE_DIR, "fiftyone_datasets/exuviae_keypoints")

CSV_FILE_SHAI = os.path.join(BASE_DIR, "fifty_one_and_analysis/measurements/exuviae/spreadsheet_files/Results-shai-exuviae.csv")

# Process Shai's CSV file
df_shai = pd.read_csv(CSV_FILE_SHAI)

# Clean up image names in Shai's data to match actual filenames
df_shai['image_name'] = df_shai['Label'].str.replace('Shai - exuviae:', '') 
df_shai['image_name'] = 'colored_' + df_shai['image_name']
# Try to load the exported dataset first
dataset_name = "prawn_keypoints"
try:
    print(f"Attempting to load dataset from {EXPORTED_DATASET_DIR}...")
    dataset = fo.Dataset.from_dir(
        dataset_dir=EXPORTED_DATASET_DIR,
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name
    )
    print(f"Successfully loaded dataset with {len(dataset)} samples")
    print("Dataset loaded with the following fields:")
    print(dataset.get_field_schema())
    
except Exception as e:
    print(f"Could not load exported dataset: {e}")
    print("Creating new dataset...")
    fo.delete_dataset("prawn_keypoints")
    dataset = fo.Dataset(dataset_name, overwrite=True)

# Set the default skeleton for the dataset
dataset.default_skeleton = fo.KeypointSkeleton(
    labels=["start_carapace", "eyes", "rostrum", "tail"],
    edges=[
        [0, 1],  # start_carapace to eyes
        [1, 2],  # eyes to rostrum
        [0, 3]   # start_carapace to tail
    ]
)

# The keypoint skeleton definition - use the one from YOLO format
KEYPOINT_SKELETON = {
    "labels": ["start_carapace", "eyes", "rostrum", "tail"],
    "edges": [
        [0, 1],  # start_carapace to eyes
        [1, 2],  # eyes to rostrum
        [0, 3]   # start_carapace to tail
    ]
}

def calculate_euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1 (tuple): The first point (x, y).
        point2 (tuple): The second point (x, y).

    Returns:
        float: The Euclidean distance between the two points.
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def process_poses(poses, is_ground_truth=False):
    """
    Process YOLO keypoint detections into FiftyOne format.
    Uses nan values for keypoints at image edges.

    Args:
        poses (list): List of YOLO pose detections
            Format: [class_id, x, y, w, h, kp1_x, kp1_y, kp1_conf, kp2_x, kp2_y, kp2_conf, ...]
        is_ground_truth (bool): Whether these are ground truth annotations

    Returns:
        tuple: (keypoints_list, detections)
            - keypoints_list: List of FiftyOne Keypoint objects
            - detections: List of FiftyOne Detection objects
    """
    keypoints_list = []
    detections = []
    EDGE_THRESHOLD = 0.01  # 1% from image edge

    for i, pose in enumerate(poses):
        if len(pose) == 17:  # 1 class + 4 bbox + 4 keypoints × 3 values
            # Bounding box processing
            x1_rel, y1_rel, width_rel, height_rel = pose[1:5]
            x1_rel -= width_rel / 2
            y1_rel -= height_rel / 2

            # Process keypoints and check for edge points
            points = []
            for i in range(5, len(pose), 3):
                x, y, conf = pose[i:i+3]
                # Check if point is at edge
                if (x < EDGE_THRESHOLD or x > 1 - EDGE_THRESHOLD or 
                    y < EDGE_THRESHOLD or y > 1 - EDGE_THRESHOLD):
                    points.append([float('nan'), float('nan')])
                else:
                    points.append([x, y])

            # Create detection first
            detection_label = "prawn_truth" if is_ground_truth else "prawn"
            detection = fo.Detection(
                label=detection_label,
                bounding_box=[x1_rel, y1_rel, width_rel, height_rel]
            )
            detections.append(detection)

            # Create keypoint object with edges and labels
            keypoint = fo.Keypoint(
                points=points,
                edges=[[0, 1], [1, 2], [0, 3]],  # Explicitly define edges
                labels=["start_carapace", "eyes", "rostrum", "tail"]  # Add labels
            )
            
            # Set keypoint attributes
            keypoint.points.attributes = {
                "color": "red",
                "radius": 8,
                "edge_color": "yellow",
                "edge_width": 2
            }
            
            # Create keypoints container
            keypoints = fo.Keypoints(keypoints=[keypoint])
            keypoints.default_attributes = {
                "color": "red",
                "radius": 8,
                "edge_color": "yellow",
                "edge_width": 2
            }
            
            # Attach keypoints to detection
            detection.keypoints = keypoints
            keypoints_list.append(keypoint)

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
    unique_lines = set()  # Track unique lines to prevent duplicates
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line_idx, line in enumerate(lines):
        # Skip duplicate lines
        if line in unique_lines:
            continue
            
        unique_lines.add(line)
        pose = [float(x) for x in line.strip().split()]
        
        # No need to reprocess coordinates, just add the raw pose
        poses.append(pose)
    
    # Print number of unique poses found
    
    return process_poses(poses)

# Load the CSV file
df = pd.read_csv(CSV_FILE)

# Process each image
processed_count = 0

for image_name in df['image_name'].unique():
    # The image_name in CSV already has the full prefix
    base_name = image_name.replace('colored_undistorted_', '')
    # Update image path to match actual file naming pattern
    image_path = os.path.join(IMAGES_DIR, f"undistorted_{base_name}.jpg")
    # Update label path to match actual file naming pattern - use the full image_name since it matches the label file
    label_path = os.path.join(LABELS_DIR, f"{image_name}.txt")
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
        
    # Check if label exists
    if not os.path.exists(label_path):
        print(f"Label not found: {label_path}")
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
    
    # Check if this image is in Shai's measurements - use the base name for matching
    shai_measurements = df_shai[df_shai['image_name'] == image_name]
    if not shai_measurements.empty:
        sample["tags"].append("shai_measured")
        # Add the number of measurements Shai made for this image
        sample["tags"].append(f"shai_measurements_{len(shai_measurements)}")
        print(f"Found Shai's measurements for {image_name}")
    
    # Process keypoints if label file exists
    if os.path.exists(label_path):
        # Use the process_poses function
        keypoints_list, detections = parse_and_process_keypoints(label_path, img_width, img_height)
        
        # Process each detection
        for detection in detections:
            # Get keypoints from the detection's keypoints field
            if not detection.keypoints or not detection.keypoints.keypoints:
                continue
                
            keypoint = detection.keypoints.keypoints[0]  # Get first keypoint from the keypoints list
            points = keypoint.points
            
            # Count points at edges (nan values)
            low_visibility_count = sum(1 for point in points if np.isnan(point[0]) or np.isnan(point[1]))
            if low_visibility_count > 0:
                sample["tags"].append(f"{low_visibility_count}_low_visibility_keypoints")
                
            # Process measurements only if required keypoints are visible
            # Points are ordered as: start_carapace, eyes, rostrum, tail
            rostrum_points = points[2]  # Index 2 is rostrum
            tail_points = points[3]     # Index 3 is tail
            
            if not (np.isnan(rostrum_points[0]) or np.isnan(rostrum_points[1]) or 
                   np.isnan(tail_points[0]) or np.isnan(tail_points[1])):
                
                # Calculate real-world measurements
                img_width_mm = 5312
                img_height_mm = 2988
                
                # Convert to real-world coordinates
                tail_points_mm = [tail_points[0] * img_width_mm, tail_points[1] * img_height_mm]
                rostrum_points_mm = [rostrum_points[0] * img_width_mm, rostrum_points[1] * img_height_mm]
                
                # Calculate distance
                tail_rostrum_distance = calculate_euclidean_distance(tail_points_mm, rostrum_points_mm)
                
                # Match with ground truth if available
                image_df = df[df['image_name'] == image_name]
                matched = False  # Track if we found a match
                for _, row in image_df.iterrows():
                    if abs(tail_rostrum_distance - float(row['pixels_total_length'])) < 30:
                        detection["total_length"] = float(row['total_length'])
                        detection['pixels_total_length'] = tail_rostrum_distance
                        
                        # Add MPE tags
                        if row['lobster_size'] == 'big':
                            mae = row['total_length'] - 180
                            if abs(mae)/180*100 < 5:
                                sample["tags"].append('big mpe<5')
                            elif abs(mae)/180*100 < 10:
                                sample["tags"].append('big mpe5<x<10')
                            elif abs(mae)/180*100 < 20:
                                sample["tags"].append('big mpe10<x<20')
                            elif abs(mae)/180*100 < 30:
                                sample["tags"].append('big mpe20<x<30')
                            else:
                                sample["tags"].append('big mpe>30')
                        elif row['lobster_size'] == 'small':
                            mae = row['total_length'] - 145
                            if abs(mae)/145*100 < 5:
                                sample["tags"].append('small mpe<5')
                            elif abs(mae)/145*100 < 10:
                                sample["tags"].append('small mpe5<x<10')
                            elif abs(mae)/145*100 < 20:
                                sample["tags"].append('small mpe10<x<20')
                            elif abs(mae)/145*100 < 30:
                                sample["tags"].append('small mpe20<x<30')
                            else:
                                sample["tags"].append('small mpe>30')
                                
                        detection['label'] = f"{row['lobster_size']}_prawn"
                        matched = True
                        break
                
                # If we went through all sizes and found no match
                if not matched:
                    print(f"Detection assigned as UNKNOWN prawn for {image_name}")
                    detection['label'] = "unknown_prawn"
            else:
                # If required keypoints are not visible
                detection['label'] = "low_visibility_prawn"
                print(f"Detection has low visibility keypoints in {image_name}")
        
        # Add keypoints and detections to sample (do this only once)
        sample["keypoints"] = fo.Keypoints(keypoints=keypoints_list)
        sample["detections"] = fo.Detections(detections=detections)
        
        # Set default keypoint options for this sample
        sample.keypoints.default_attributes = {"color": "red", "radius": 8, "edge_color": "yellow", "edge_width": 2}
        for keypoint in keypoints_list:
            keypoint.points.attributes = {"color": "red", "radius": 8, "edge_color": "yellow", "edge_width": 2}
            
    # Set a meaningful name for display in the app
    sample["name"] = f"{os.path.splitext(os.path.basename(image_path))[0]} - {pond_type}"
    

    # add bounding box to sample from df_shai
    shai_df = df_shai[df_shai['image_name'] == image_name]
    shai_detections = []
    shai_polyline1 = []
    shai_polyline2 = []
    if not shai_df.empty:
        for _,row in shai_df.iterrows():
            #convert BX, BY, Width, Height to int
            BX = row['BX']
            BY = row['BY']
            Width = row['Width']
            Height = row['Height']

            img_width_mm = 5312
            img_height_mm = 2988

            #add polylines of diagonal of bounding box
            bounding_box =[BX/img_width_mm, BY/img_height_mm, Width/img_width_mm, Height/img_width_mm] 
            #add polylines of diagonal of bounding box
            

            top_left_max = [BX/img_width_mm, BY/img_height_mm]
            top_right_max = [BX/img_width_mm + Width/img_width_mm, BY/img_height_mm]
            bottom_left_max = [BX/img_width_mm, BY/img_height_mm + Height/img_height_mm]
            bottom_right_max = [BX/img_width_mm + Width/img_width_mm, BY/img_height_mm + Height/img_height_mm]

            # Diagonals
            diagonal1_max = [top_left_max, bottom_right_max]
            diagonal2_max = [top_right_max, bottom_left_max]


            diagonal_polyline1 = fo.Polyline(points=[diagonal1_max])
            diagonal_polyline2 = fo.Polyline(points=[diagonal2_max])
            shai_polyline1.append(diagonal_polyline1)
            shai_polyline2.append(diagonal_polyline2)
            
            shai_detections.append(fo.Detection(bounding_box=bounding_box,label=str(row['Length'])))
    sample["shai_polyline1"] = fo.Polylines(polylines=shai_polyline1)
    sample["shai_polyline2"] = fo.Polylines(polylines=shai_polyline2)

    sample["bounding_box"] = fo.Detections(detections=shai_detections)
      
        

       
        # sample["bounding_box"] = fo.BoundingBox(
        #     x=shai_row['BX'],
        #     y=shai_row['BY'],
        #     width=shai_row['Width'],
        #     height=shai_row['Height']
   

    # Add sample to dataset
    dataset.add_sample(sample)
    processed_count += 1
    
    if processed_count % 10 == 0:
        print(f"Processed {processed_count} images")

print(f"Dataset created with {len(dataset)} samples")
print("You can filter samples with low visibility keypoints using: dataset.match_tags('*_low_visibility_keypoints')")
print("You can now view the dataset with: dataset.app()")

# Create some views for analysis
try:
    # Create views based on pond type
    circle_pond_view = dataset.match_tags("Circle")
    square_pond_view = dataset.match_tags("Square")
    low_visibility_view = dataset.match_tags("*_low_visibility_keypoints")
    shai_measured_view = dataset.match_tags("shai_measured")
    
    print(f"Found {len(circle_pond_view)} samples from Circle pond")
    print(f"Found {len(square_pond_view)} samples from Square pond")
    print(f"Found {len(low_visibility_view)} samples with low visibility keypoints")
    print(f"Found {len(shai_measured_view)} samples measured by Shai")
except Exception as e:
    print(f"Error creating views: {str(e)}")

# Save the dataset
dataset.persistent = True
dataset.save() 

def export_dataset(dataset, export_dir):
    """
    Export the dataset to the specified directory in FiftyOne format.
    
    Args:
        dataset: FiftyOne dataset to export
        export_dir: Directory to export the dataset to
    """
    print(f"Exporting dataset to {export_dir}...")
    try:
        # Export the dataset
        dataset.export(
            export_dir=export_dir,
            dataset_type=fo.types.FiftyOneDataset,
            export_media=True  # Set to True to copy images to export directory
        )
        print(f"Successfully exported dataset to {export_dir}")
    except Exception as e:
        print(f"Error exporting dataset: {e}")

# After creating the dataset, add a default view to show keypoints
print("Setting up default view options for keypoints display...")

try:
    # Configure FiftyOne to use headless mode for more stable behavior
    fo.config.show_progress_bars = True
    
    # Launch the app but don't wait for it
    print("Launching FiftyOne app with dataset...")
    port = random.randint(5151, 5200)
    session = fo.launch_app(dataset, port=port)
    
    # Configure the view to show keypoints properly
    session.view = dataset.view()
    
    # Set default keypoint display options using the new style
    session.view.config.keypoints.show_edges = True  # Show edges between keypoints
    session.view.config.keypoints.show_points = True  # Show keypoint points
    session.view.config.keypoints.show_labels = True  # Show keypoint labels
    session.view.config.keypoints.edge_color = "yellow"  # Set edge color
    session.view.config.keypoints.edge_width = 2  # Set edge width
    session.view.config.keypoints.point_color = "red"  # Set point color
    session.view.config.keypoints.point_size = 8  # Set point radius
    
    

    # Set default detection display options
    print("Dataset is ready to view in the app")
    
except Exception as e:
    print(f"Error launching app: {e}")

if __name__ == "__main__":
    """
    Main execution block for the script. It provides options to view, load, and export the dataset.
    """
    print("\nDataset is ready to view. You can:")
    print("1. Access it in the browser at http://localhost:5151")
    print("2. Load it in Python with: fo.load_dataset('prawn_keypoints')")
    print("3. Export it to a new directory with: export_dataset(dataset, 'path/to/export')")
    
    # Export the dataset to the default export directory
    export_dir = os.path.join(BASE_DIR, "exported_datasets/exuviae_keypoints")
    os.makedirs(export_dir, exist_ok=True)
    export_dataset(dataset, export_dir)
    
    port = random.randint(5151, 5200)
    session = fo.launch_app(dataset, port=port)
    session.wait()