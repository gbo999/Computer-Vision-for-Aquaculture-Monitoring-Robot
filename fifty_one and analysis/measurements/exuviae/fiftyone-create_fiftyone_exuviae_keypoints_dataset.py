import os
import pandas as pd
import numpy as np
import fiftyone as fo
import fiftyone.utils.data as foud
from fiftyone import ViewField as F
import glob
from PIL import Image  # Add PIL for image operations

# Use absolute paths for everything to avoid path issues
BASE_DIR = "/Users/gilbenor/Documents/code_projects/msc/counting_research_algorithms"
LABELS_DIR = os.path.join(BASE_DIR, "training and val output/runs/pose/predict83/labels")
IMAGES_DIR = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized"
CSV_FILE = os.path.join(BASE_DIR, "fifty_one and analysis/measurements/exuviae/spreadsheet_files/length_analysis_new_split.csv")
EXPORTED_DATASET_DIR = os.path.join(BASE_DIR, "fiftyone_datasets/exuviae_keypoints")

CSV_FILE_SHAI = os.path.join(BASE_DIR, "fifty_one and analysis/measurements/exuviae/spreadsheet_files/Results-shai-exuviae.csv")

df_shai = pd.read_csv(CSV_FILE_SHAI)

df_shai['image_name'] = df_shai['Label'].str.replace('Shai - exuviae:', 'colored_')

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









# The keypoint skeleton definition - use the one from YOLO formatxs
dataset.default_skeleton = fo.KeypointSkeleton(
    labels=["start_carapace", "eyes", "rostrum", "tail"],
    edges=[
        [0, 1],  # start_carapace to eyes
        [1, 2],  # eyes to rostrum
        [0, 3]   # start_carapace to tail
    ]
)
def calculate_euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


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

    for i, pose in enumerate(poses):
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

    #get row from df
    
    # Construct paths - the file is already prefixed with 'colored_'
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
        img_width_mm = 5312
        img_height_mm = 2988
        for detection in detections:
            # Properly scale x and y coordinates separately
            # Assuming keypoint.points is a flat list [x1, y1, x2, y2, ...]
            #coords of keypoints are in [x, y] format
            #scale x and y coordinates separately for keypoints
            # Get keypoints from detection attributes
            keypoints_dict = detection.attributes["keypoints"]
            tail_points = keypoints_dict["tail"]
            rostrum_points = keypoints_dict["rostrum"]
            eyes_points = keypoints_dict["eyes"]
            start_carapace_points = keypoints_dict["start_carapace"]

            #multiply x and y coordinates by img_width_mm and img_height_mm to get real world coordinates
            tail_points = [tail_points[0] * img_width_mm, tail_points[1] * img_height_mm]
            rostrum_points = [rostrum_points[0] * img_width_mm, rostrum_points[1] * img_height_mm]
            eyes_points = [eyes_points[0] * img_width_mm, eyes_points[1] * img_height_mm]
            start_carapace_points = [start_carapace_points[0] * img_width_mm, start_carapace_points[1] * img_height_mm]


            #compute pixels between rostrum and tail
            rostrum_tail_distance = calculate_euclidean_distance(rostrum_points, tail_points)

            # Calculate distance between tail and rostrum keypoints
            tail_rostrum_distance = calculate_euclidean_distance(tail_points, rostrum_points)

            # Calculate distance between tail and eyes keypoints

            # Get unique lobster sizes for this image
            image_df = df[df['image_name'] == image_name]
            matched = False
            
            # First, try to match with each unique lobster size
            for lobster_size in image_df['lobster_size'].unique():
                # Get row for this lobster size
                size_rows = image_df[image_df['lobster_size'] == lobster_size]
                if len(size_rows) == 0:
                    continue
                    
                row = size_rows.iloc[0]  # Use the first row for this size
                
                # Check if this detection is a match for this row based on distance
                is_match = abs(tail_rostrum_distance - float(row['pixels_total_length'])) < 30
                
                if is_match:
                    detection["total_length"] = float(row['total_length'])
                    detection['pixels_total_length'] = tail_rostrum_distance
                    # detection["carapace_length"] = float(row['carapace_length']) if not pd.isna(row['total_length']) else None
                    # sample["tags"].append(f"{lobster_size}_prawn_{row['total_length']:.1f}mm")
                    # sample["tags"].append(f"{lobster_size}_prawn_{row['carapace_length']:.1f}mm")
                    
                    
                    if row['lobster_size'] == 'big':
                        mae=row['total_length']-180
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
                        mae=row['total_length']-145
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
                            
                    detection['label'] = f"{lobster_size}_prawn"
                    matched = True
                    shai_df = df_shai[df_shai['image_name'] == image_name]
                    for _,row in shai_df.iterrows():
                        if abs(row['Length']-tail_rostrum_distance)/row['Length']*100 < 10:
                            sample["tags"].append('<10')
                        elif abs(row['Length']-tail_rostrum_distance)/row['Length']*100 < 20:
                            sample["tags"].append('10<x<20')
                        elif abs(row['Length']-tail_rostrum_distance)/row['Length']*100 < 30:
                            sample["tags"].append('20<x<30')
                        else:
                            sample["tags"].append('>30')




                    break
            
            # If we went through all sizes and found no match
            if not matched:
                print(f"detection assigned as UNKNOWN prawn for {image_name}")
            # Add keypoints and detections to sample
        sample["keypoints"] = fo.Keypoints(keypoints=keypoints_list)
        sample["detections"] = fo.Detections(detections=detections)
    
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
            bounding_box =[BX/img_width_mm, BY/img_height_mm, Width/img_width_mm, Height/img_height_mm] 
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

try:
    # Configure FiftyOne to use headless mode for more stable behavior
    fo.config.show_progress_bars = True
    
    # Launch the app but don't wait for it
    print("Launching FiftyOne app with dataset...")
    session = fo.launch_app(dataset, port=5151)
    
    # Configure the view to show keypoints properly
    session.view = dataset.view()
    session.config.show_confidence = True
    session.config.show_attributes = True
    session.config.show_keypoints = True
    
    # Only export if we created a new dataset
    if not os.path.exists(EXPORTED_DATASET_DIR):
        print(f"Exporting dataset to {EXPORTED_DATASET_DIR}...")
        dataset.export(
            export_dir=EXPORTED_DATASET_DIR,
            dataset_type=fo.types.FiftyOneDataset,
            export_media=True
        )
        print("Dataset exported successfully")

    # Save the session configuration
    session.wait()
    
    print("View configured to show keypoints. Access the app in your browser.")
    print(f"App URL: http://localhost:5151")
    
except Exception as e:
    print(f"Error launching app: {e}")

if __name__ == "__main__":
    print("\nDataset is ready to view. You can:")
    print("1. Access it in the browser at http://localhost:5151")
    print("2. Load it in Python with: fo.load_dataset('prawn_keypoints')")
    print("3. Export it again with: dataset.export(export_dir='path/to/export')")
    session = fo.launch_app(dataset)
    session.wait()