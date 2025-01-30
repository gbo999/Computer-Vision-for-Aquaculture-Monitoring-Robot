import os
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_and_filter_labels(labels_dir, confidence_df):
    """
    Filter YOLO detections based on confidence thresholds from Excel sheet.
    
    Args:
        labels_dir (str): Path to directory containing YOLO label files
        confidence_df (pd.DataFrame): DataFrame with columns ['image_name', 'confidence']
    
    Returns:
        dict: Statistics about filtered detections
    """
    # Create DataFrame for storing detailed stats
    detailed_stats = pd.DataFrame(columns=['image_name', 'target_confidence', 'object_confidence', 'object_poses', 'detection', 'which'])
    
    stats = {
        "total_detections": 0,
        "filtered_confidence": 0,
        "files_processed": 0,
        "files_modified": 0
    }
    
    # Process each label file
    for label_file in Path(labels_dir).glob("*.txt"):
        base_name = label_file.stem
        print(f"\n\nProcessing file: {base_name}")
        
        # Print Excel row match
        image_row = confidence_df[confidence_df['image_name'].str.contains(base_name)]
        print(f"Matching Excel row:\n{image_row}")
        
        # Print label file contents
        print("Label file contents:")
        with open(label_file, 'r') as f:
                
        # Find matching confidence threshold
         image_row = confidence_df[confidence_df['image_name'].str.contains(base_name)]
        if image_row.empty:
            print(f"No confidence threshold found for {base_name}, skipping...")
            continue
            
        target_confidence = float(image_row['confidence'].iloc[0])
        which_value = image_row['which'].iloc[0]  # Get the 'which' value for this specific image
        
        modified = False
        valid_detections = []
        
        # Read detections
        with open(label_file, 'r') as f:
            detections = f.readlines()
        
        stats["files_processed"] += 1
        stats["total_detections"] += len(detections)
        
        # Process each detection
        for det in detections:
            values = list(map(float, det.strip().split()))
            
                # Get object confidence from YOLO format
            obj_confidence = values[-1]  # Confidence score is typically after bbox coordinates
            # Keep detection if confidence matches the target
           

            print(f"target_confidence: {target_confidence}, obj_confidence: {obj_confidence}")
            if abs(obj_confidence - target_confidence) < 0.1:  # Using small epsilon for float comparison
                valid_detections.append(det)
                
                # Add row to detailed stats
                detailed_stats = pd.concat([detailed_stats, pd.DataFrame({
                    'image_name': [base_name],
                    'target_confidence': [target_confidence],
                    'object_confidence': [obj_confidence],
                    'object_poses': [values[5:]],
                    'detection': [det.strip()],
                    'which': [which_value]
                })], ignore_index=True)
            else:
                stats["filtered_confidence"] += 1
        
        # Save filtered detections if any were removed
        # if len(valid_detections) != len(detections):
        #     modified = True
        #     stats["files_modified"] += 1
        #     with open(label_file, 'w') as f:
        #         f.writelines(valid_detections)
        
        # Print progress for current file
        print(f"Processed {label_file.name}: {len(detections)} -> {len(valid_detections)} detections (target conf: {target_confidence})")
    
    # Save detailed stats to CSV
    detailed_stats.to_csv("/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/measurement_analysis_square.csv", index=False)
    
    # Print summary statistics
    print("\nFiltering Summary:")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files modified: {stats['files_modified']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Filtered by confidence: {stats['filtered_confidence']}")
    print(f"Remaining detections: {stats['total_detections'] - stats['filtered_confidence']}")
    
    return stats

# Read confidence thresholds from Excel
confidence_df = pd.read_excel("/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/molts_all.xlsx")

# Process the labels
labels_dir = "/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/runs/predict/square_molt/exp/labels"
stats = analyze_and_filter_labels(labels_dir, confidence_df)
