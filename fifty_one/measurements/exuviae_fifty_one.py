import fiftyone as fo
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

def create_dataset():
    """Create and configure the FiftyOne dataset with keypoint skeleton"""
    dataset = fo.Dataset("molt_filtered_dataset", overwrite=True, persistent=True)
    dataset.default_skeleton = fo.KeypointSkeleton(
        labels=["start_carapace", "eyes", "rostrum", "tail"],
        edges=[
            [0, 1],  # start_carapace to eyes
            [1, 2],  # eyes to rostrum
            [0, 3]   # start_carapace to tail
        ]
    )
    return dataset

def calculate_error_tag(measured, expected):
    """Calculate error percentage and return appropriate tag"""
    if pd.isna(measured):
        return None
    
    error_percent = abs(measured - expected) / expected * 100
    
    if error_percent < 5:
        return "error_below_5"
    elif error_percent < 10:
        return "error_5_to_10"
    else:
        return "error_above_10"

def process_label_file(label_path):
    """Read and process a label file to extract all keypoints"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return []
            
        # Process all detections in the file
        all_keypoints = []
        for line in lines:
            values = list(map(float, line.strip().split()))
            
            keypoints = []
            for i in range(5, len(values)-1, 3):
                x = values[i]
                y = values[i+1]
                keypoints.append([x, y])
                
            all_keypoints.append(keypoints)
            
        return all_keypoints

def main():
    # Define paths
    images_dir = Path("/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized")
    analysis_csv = Path("runs/pose/predict57/length_analysis.csv")
    labels_dir = Path("runs/pose/predict57/further_labels_files")
    
    # Expected values
    expected_big_total = 180  # mm
    expected_small_total = 145  # mm
    
    # Read the analysis CSV
    analysis_df = pd.read_csv(analysis_csv)
    
    # Create dataset
    dataset = create_dataset()
    
    # Process each image in the analysis CSV
    for _, row in tqdm(analysis_df.iterrows(), desc="Processing images"):
        image_name = row['image_name']
        image_path = images_dir / image_name
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        label_file = labels_dir / f"{image_name.replace('.jpg', '.txt')}"
        if not label_file.exists():
            print(f"Warning: Label file not found: {label_file}")
            continue
            
        all_keypoints = process_label_file(label_file)
        if not all_keypoints:
            print(f"Warning: No detections in label file: {label_file}")
            continue
        
        # Create sample
        sample = fo.Sample(filepath=str(image_path))
        
        # Initialize tags list
        tags = []
        
        # Calculate error tags
        big_error_tag = calculate_error_tag(row['big_total_length'], expected_big_total)
        small_error_tag = calculate_error_tag(row['small_total_length'], expected_small_total)
        
        # Add tags if they exist
        if big_error_tag:
            tags.append(f"big_{big_error_tag}")
        if small_error_tag:
            tags.append(f"small_{small_error_tag}")
        
        # Add tags to sample
        sample.tags = tags
        
        # Add keypoints field to sample
        keypoints_list = []
        
        # Add big prawn keypoints if available
        if pd.notna(row['big_total_length']) and len(all_keypoints) > 0:
            error_percent = abs(row['big_total_length'] - expected_big_total) / expected_big_total * 100
            keypoints_list.append(
                fo.Keypoint(
                    points=all_keypoints[0],
                    label=f"BIG: Total={row['big_total_length']:.1f}mm (Error: {error_percent:.1f}%)"
                )
            )
        
        # Add small prawn keypoints if available
        if pd.notna(row['small_total_length']) and len(all_keypoints) > 1:
            error_percent = abs(row['small_total_length'] - expected_small_total) / expected_small_total * 100
            keypoints_list.append(
                fo.Keypoint(
                    points=all_keypoints[1],
                    label=f"SMALL: Total={row['small_total_length']:.1f}mm (Error: {error_percent:.1f}%)"
                )
            )
        
        # If we have keypoints to add
        if keypoints_list:
            keypoints_obj = fo.Keypoints(
                keypoints=keypoints_list,
                skeleton=dataset.default_skeleton
            )
            sample["keypoints"] = keypoints_obj
        
            # Add measurements as metadata
            sample["big_length"] = row['big_total_length']
            sample["small_length"] = row['small_total_length']
            
            # Add error percentages as metadata
            if pd.notna(row['big_total_length']):
                sample["big_error_percent"] = abs(row['big_total_length'] - expected_big_total) / expected_big_total * 100
            if pd.notna(row['small_total_length']):
                sample["small_error_percent"] = abs(row['small_total_length'] - expected_small_total) / expected_small_total * 100
            
            # Add the sample to the dataset
            dataset.add_sample(sample)
    
    # Print summary of tags
    print("\nDataset Statistics:")
    print(f"Total samples: {len(dataset)}")
    print("\nError Distribution:")
    print("Big prawns:")
    print(f"  Below 5%: {len(dataset.match_tags('big_error_below_5'))}")
    print(f"  5-10%: {len(dataset.match_tags('big_error_5_to_10'))}")
    print(f"  Above 10%: {len(dataset.match_tags('big_error_above_10'))}")
    print("\nSmall prawns:")
    print(f"  Below 5%: {len(dataset.match_tags('small_error_below_5'))}")
    print(f"  5-10%: {len(dataset.match_tags('small_error_5_to_10'))}")
    print(f"  Above 10%: {len(dataset.match_tags('small_error_above_10'))}")
    
    # Launch the FiftyOne app
    session = fo.launch_app(dataset, port=5159)
    session.wait()

if __name__ == "__main__":
    main()