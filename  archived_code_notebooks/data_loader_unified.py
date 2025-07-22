import fiftyone as fo
import pandas as pd
import os
import ast
from tqdm import tqdm
from utils import parse_pose_estimation, calculate_euclidean_distance, calculate_real_width, extract_identifier_from_gt, calculate_bbox_area

def create_unified_dataset(weights_type):
    """Create a unified FiftyOne dataset for both carapace and body measurements."""
    dataset_name = f"prawn_dataset_unified_{weights_type}"
    dataset_dir = f"fiftyone_datasets/unified_{weights_type}"
    
    # Check if dataset exists
    if os.path.exists(dataset_dir):
        try:
            dataset = fo.Dataset.from_dir(
                dataset_dir=dataset_dir,
                dataset_type=fo.types.FiftyOneDataset,
                name=dataset_name
            )
            return dataset, True
        except Exception as e:
            print(f"Error loading dataset from {dataset_dir}: {e}")
    
    # Create new dataset
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    
    # Set up skeleton for unified measurements
    dataset.default_skeleton = fo.KeypointSkeleton(
        labels=["start_carapace", "eyes", "rostrum", "tail"],
        edges=[
            [0, 1],  # start_carapace to eyes
            [1, 2],  # eyes to rostrum
            [0, 3]   # start_carapace to tail
        ]
    )
    return dataset, False

def process_poses(poses, is_ground_truth=False):
    """Process YOLO keypoint detections into FiftyOne format."""
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
                'start_carapace': keypoints[0],
                'eyes': keypoints[1],
                'rostrum': keypoints[2],
                'tail': keypoints[3],
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
                    label="prawn_truth",
                    bounding_box=[x1_rel, y1_rel, width_rel, height_rel],
                    attributes={'keypoints': keypoints_dict}
                ))
    
    return keypoints_list, detections

def process_unified_images(image_paths, prediction_folder_path, ground_truth_paths_text, carapace_df, body_df, metadata_df, dataset, pond_type):
    """Process images with both carapace and body measurements."""
    for image_path in tqdm(image_paths):
        identifier = extract_identifier_from_gt(image_path)
        if not identifier:
            print(f"Warning: Could not extract identifier from {image_path}")
            continue
        
        # Find prediction file
        prediction_txt_path = None
        for pred_file in os.listdir(prediction_folder_path):
            if identifier in pred_file:
                prediction_txt_path = os.path.join(prediction_folder_path, pred_file)
                break
        
        # Find ground truth file
        ground_truth_txt_path = None
        for gt_file in ground_truth_paths_text:
            b = extract_identifier_from_gt(gt_file)
            if b == identifier:
                ground_truth_txt_path = gt_file
                break
        
        if ground_truth_txt_path is None:
            print(f"No ground truth found for {identifier}")
            continue
        
        # Process poses
        pose_estimations = parse_pose_estimation(prediction_txt_path)
        ground_truths = parse_pose_estimation(ground_truth_txt_path)
        
        keypoints_list, detections = process_poses(pose_estimations)
        keypoints_list_truth, detections_truth = process_poses(ground_truths, is_ground_truth=True)
        
        # Create sample
        sample = fo.Sample(filepath=image_path)
        sample["ground_truth"] = fo.Detections(detections=detections_truth)
        sample["detections_predictions"] = fo.Detections(detections=detections)
        sample["keypoints"] = fo.Keypoints(keypoints=keypoints_list)
        sample["keypoints_truth"] = fo.Keypoints(keypoints=keypoints_list_truth)
        sample.tags.append(pond_type)
        
        # Add metadata from both carapace and body measurements
        add_unified_metadata(sample, identifier, carapace_df, body_df, metadata_df)
        
        # Add sample to dataset
        dataset.add_sample(sample)
    
    return carapace_df, body_df

def add_unified_metadata(sample, identifier, carapace_df, body_df, metadata_df):
    """Add metadata from both carapace and body measurements."""
    # Find matching rows in both dataframes
    carapace_rows = carapace_df[carapace_df['Label'].str.contains(identifier, na=False)]
    body_rows = body_df[body_df['Label'].str.contains(identifier, na=False)]
    
    if carapace_rows.empty and body_rows.empty:
        print(f"No matching rows found for {identifier}")
        return
    
    # Get filename from either dataframe
    filename = None
    if not carapace_rows.empty:
        filename = carapace_rows['Label'].values[0].split(':')[1]
    elif not body_rows.empty:
        filename = body_rows['Label'].values[0].split(':')[1]
    
    # Create metadata lookup key
    parts = identifier.split('_')
    relevant_part = f"{parts[0]}_{parts[-1]}"
    
    # Add camera metadata
    metadata_row = metadata_df[metadata_df['file_name_new'] == relevant_part]
    if not metadata_row.empty:
        metadata = metadata_row.iloc[0].to_dict()
        for key, value in metadata.items():
            if key != 'file name':
                sample[key] = value
    else:
        print(f"No metadata found for {relevant_part}")
    
    # Process carapace detections
    if not carapace_rows.empty:
        for _, row in carapace_rows.iterrows():
            process_detection_row(sample, row, "carapace")
    
    # Process body detections
    if not body_rows.empty:
        for _, row in body_rows.iterrows():
            process_detection_row(sample, row, "body")

def process_detection_row(sample, row, measurement_type):
    """Process a single detection row from either carapace or body measurements."""
    prawn_id = row['PrawnID']
    
    # Process each bounding box
    for bbox_key in ['BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3']:
        if pd.notna(row[bbox_key]):
            bbox = ast.literal_eval(row[bbox_key])
            closest_detection, ground = find_closest_detection(sample, bbox)
            if closest_detection is not None:
                # Add metadata to detection
                closest_detection.attributes['prawn_id'] = prawn_id
                closest_detection.attributes['measurement_type'] = measurement_type
                for col in row.index:
                    if col not in ['Label', 'PrawnID', 'BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3']:
                        closest_detection.attributes[col] = row[col] 