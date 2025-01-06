import os
import numpy as np
import pandas as pd
from pathlib import Path
import ast

def string_to_list(list_str):
    """
    Convert string representation of a list to an actual list of floats.
    Example: "[0.758122, 0.86973, 0.999739, ...]" -> [0.758122, 0.86973, 0.999739, ...]
    """
    # Remove brackets and split by comma
    clean_str = list_str.strip('[]')
    # Split by comma and convert each value to float
    return [float(x.strip()) for x in clean_str.split(',')]

def analyze_and_filter_labels(labels_dir, confidence_df):
    """
    Filter YOLO detections based on confidence thresholds from Excel sheet.
    """
    # Create DataFrame for storing detailed stats
    detailed_stats = pd.DataFrame(columns=['image_name', 'target_confidence', 'object_confidence', 'object_poses', 'detection'])
    
    stats = {
        "total_detections": 0,
        "filtered_confidence": 0,
        "files_processed": 0,
        "files_modified": 0
    }
    
    # Process each label file
    for label_file in Path(labels_dir).glob("*.txt"):
        base_name = label_file.stem
        
        # Find matching confidence threshold
        image_row = confidence_df[confidence_df['image_name'].str.contains(base_name)]
        if image_row.empty:
            print(f"No confidence threshold found for {base_name}, skipping...")
            continue
            
        target_confidence = float(image_row['confidence'].iloc[0])
        valid_detections = []
        
        # Read detections
        with open(label_file, 'r') as f:
            detections = f.readlines()
        
        stats["files_processed"] += 1
        stats["total_detections"] += len(detections)
        
        # Process each detection
        for det in detections:
            values = list(map(float, det.strip().split()))
            obj_confidence = values[1]  # Confidence is second value in YOLO format
            
            if abs(obj_confidence - target_confidence) < 0.01:
                valid_detections.append(det)
                
                # Get poses as list
                poses = values[5:]  # Get all pose values
                
                # Add row to detailed stats
                detailed_stats = pd.concat([detailed_stats, pd.DataFrame({
                    'image_name': [base_name],
                    'target_confidence': [target_confidence],
                    'object_confidence': [obj_confidence],
                    'object_poses': [poses],  # Store as actual list
                    'detection': [det.strip()]
                })], ignore_index=True)
            else:
                stats["filtered_confidence"] += 1
        
        print(f"Processed {label_file.name}: {len(detections)} -> {len(valid_detections)} detections")
    
    # Save detailed stats to CSV
    detailed_stats.to_csv("fifty_one/measurements/filter_detections_stats_right.csv", index=False)
    
    return stats, detailed_stats

# Example usage:
confidence_df = pd.read_excel("/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/molts.xlsx")
labels_dir = "/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/runs/pose/predict19/labels"
stats, df = analyze_and_filter_labels(labels_dir, confidence_df)

# Example of converting a string to list
example_str = "[0.758122, 0.86973, 0.999739, 0.728687, 0.881113, 0.999844, 0.707809, 0.885143, 0.999248, 0.827013, 0.836563, 0.997457, 0.758438]"
poses_list = string_to_list(example_str)
print("Converted list:", poses_list)
