import fiftyone as fo
import os
import pandas as pd
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

def process_label_file(label_path):
    """Read and process a label file to extract all keypoints"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return []
            
        # Process all detections in the file (not just the first one)
        all_keypoints = []
        for line in lines:
            values = list(map(float, line.strip().split()))
            
            # Extract keypoints (they start at index 5, in groups of 3: x,y,conf)
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
    
    # Read the analysis CSV
    analysis_df = pd.read_csv(analysis_csv)
    
    # Create dataset
    dataset = create_dataset()
    
    # Process each image in the analysis CSV
    for _, row in tqdm(analysis_df.iterrows(), desc="Processing images"):
        image_name = row['image_name']
        image_path = images_dir / image_name
        
        # Check if image exists
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get corresponding label file
        label_file = labels_dir / f"{image_name.replace('.jpg', '.txt')}"
        if not label_file.exists():
            print(f"Warning: Label file not found: {label_file}")
            continue
            
        # Process label file to get all keypoints
        all_keypoints = process_label_file(label_file)
        if not all_keypoints:
            print(f"Warning: No detections in label file: {label_file}")
            continue
        
        # Create sample
        sample = fo.Sample(filepath=str(image_path))
        
        # Add keypoints field to sample
        keypoints_list = []
        
        # Add big prawn keypoints if available
        if pd.notna(row['big_total_length']) and len(all_keypoints) > 0:
            keypoints_list.append(
                fo.Keypoint(
                    points=all_keypoints[0],  # First detection is likely the big prawn
                    label=f"BIG: Total={row['big_total_length']:.1f}mm, Carapace={row['big_carapace_length']:.1f}mm"
                )
            )
        
        # Add small prawn keypoints if available
        if pd.notna(row['small_total_length']) and len(all_keypoints) > 1:
            keypoints_list.append(
                fo.Keypoint(
                    points=all_keypoints[1],  # Second detection is likely the small prawn
                    label=f"SMALL: Total={row['small_total_length']:.1f}mm, Carapace={row['small_carapace_length']:.1f}mm"
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
            
            # Add the sample to the dataset
            dataset.add_sample(sample)
    
    print(f"Created dataset with {len(dataset)} samples")
    
    # Launch the FiftyOne app
    session = fo.launch_app(dataset, port=5159)
    session.wait()

if __name__ == "__main__":
    main()