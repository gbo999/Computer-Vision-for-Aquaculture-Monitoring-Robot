import pandas as pd
import numpy as np
import os

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    # box format: [x1, y1, width, height]
    
    # Convert to [x1, y1, x2, y2] format
    box1_x2 = box1[0] + box1[2]
    box1_y2 = box1[1] + box1[3]
    box2_x2 = box2[0] + box2[2]
    box2_y2 = box2[1] + box2[3]
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersection
    
    return intersection / union

def get_bounding_box_from_keypoints(row):
    """Calculate bounding box from keypoints"""
    # Get all x and y coordinates
    x_coords = [row['start_carapace_x'], row['eyes_x'], row['rostrum_x'], row['tail_x']]
    y_coords = [row['start_carapace_y'], row['eyes_y'], row['rostrum_y'], row['tail_y']]
    
    # Remove NaN values
    x_coords = [x for x in x_coords if not np.isnan(x)]
    y_coords = [y for y in y_coords if not np.isnan(y)]
    
    if not x_coords or not y_coords:
        return None
    
    # Calculate bounding box
    x1 = min(x_coords)
    y1 = min(y_coords)
    width = max(x_coords) - x1
    height = max(y_coords) - y1
    
    return [x1, y1, width, height]

# Load the datasets
manual_csv = "spreadsheet_files/manual_classifications_with_locations.csv"
shai_csv = "../../../fifty_one_and_analysis/measurements/exuviae/spreadsheet_files/Results-shai-exuviae.csv"

df_manual = pd.read_csv(manual_csv)
df_shai = pd.read_csv(shai_csv)

print(f"Loaded {len(df_manual)} manual classifications")
print(f"Loaded {len(df_shai)} Shai measurements")

# Clean up image names in Shai's data
df_shai['image_name'] = df_shai['Label'].str.replace('Shai - exuviae:', '')
df_shai['image_name'] = 'colored_' + df_shai['image_name']

# Convert Shai's bounding boxes to normalized coordinates
img_width_mm = 5312
img_height_mm = 2988

df_shai['BX_norm'] = df_shai['BX'] / img_width_mm
df_shai['BY_norm'] = df_shai['BY'] / img_height_mm
df_shai['Width_norm'] = df_shai['Width'] / img_width_mm
df_shai['Height_norm'] = df_shai['Height'] / img_height_mm

# Calculate bounding boxes for manual detections
df_manual['bbox'] = df_manual.apply(get_bounding_box_from_keypoints, axis=1)

# Prepare merged dataframe
merged_data = []

# For each manual detection
for idx_manual, row_manual in df_manual.iterrows():
    bbox_manual = row_manual['bbox']
    if bbox_manual is None:
        continue
        
    # Get corresponding image in Shai's data
    shai_image_rows = df_shai[df_shai['image_name'] == row_manual['image_name']]
    
    best_match = None
    best_iou = 0
    
    # Find the best matching bounding box from Shai's measurements
    for idx_shai, row_shai in shai_image_rows.iterrows():
        bbox_shai = [row_shai['BX_norm'], row_shai['BY_norm'], 
                    row_shai['Width_norm'], row_shai['Height_norm']]
        
        iou = calculate_iou(bbox_manual, bbox_shai)
        
        if iou > best_iou:
            best_iou = iou
            best_match = row_shai
    
    # If we found a match with sufficient overlap
    if best_iou > 0.5:  # You can adjust this threshold
        merged_row = {
            'image_name': row_manual['image_name'],
            'manual_size': row_manual['manual_size'],
            'shai_length': best_match['Length'],
            'tail_rostrum_distance_mm': row_manual['tail_rostrum_distance_mm'],
            'iou_score': best_iou,
            # Add all keypoint coordinates
            'start_carapace_x': row_manual['start_carapace_x'],
            'start_carapace_y': row_manual['start_carapace_y'],
            'eyes_x': row_manual['eyes_x'],
            'eyes_y': row_manual['eyes_y'],
            'rostrum_x': row_manual['rostrum_x'],
            'rostrum_y': row_manual['rostrum_y'],
            'tail_x': row_manual['tail_x'],
            'tail_y': row_manual['tail_y'],
            # Add Shai's bounding box
            'shai_BX': best_match['BX'],
            'shai_BY': best_match['BY'],
            'shai_Width': best_match['Width'],
            'shai_Height': best_match['Height']
        }
        merged_data.append(merged_row)

# Create merged DataFrame
df_merged = pd.DataFrame(merged_data)

# Calculate statistics
print("\nMerging Statistics:")
print(f"Total manual detections: {len(df_manual)}")
print(f"Total Shai measurements: {len(df_shai)}")
print(f"Successfully merged: {len(df_merged)}")
print(f"Average IoU score: {df_merged['iou_score'].mean():.3f}")

# Add analysis columns
df_merged['length_difference'] = df_merged['shai_length'] - df_merged['tail_rostrum_distance_mm']
df_merged['length_difference_percentage'] = (df_merged['length_difference'] / df_merged['shai_length']) * 100

# Print analysis
print("\nLength Difference Analysis:")
print(f"Mean difference: {df_merged['length_difference'].mean():.2f} mm")
print(f"Mean percentage difference: {df_merged['length_difference_percentage'].mean():.2f}%")
print(f"Standard deviation: {df_merged['length_difference'].std():.2f} mm")

# Export merged data
output_path = "spreadsheet_files/merged_manual_and_shai.csv"
df_merged.to_csv(output_path, index=False)
print(f"\nMerged data exported to: {output_path}")

# Create separate files for big and small prawns
df_big = df_merged[df_merged['manual_size'] == 'big']
df_small = df_merged[df_merged['manual_size'] == 'small']

df_big.to_csv("spreadsheet_files/merged_manual_and_shai_big.csv", index=False)
df_small.to_csv("spreadsheet_files/merged_manual_and_shai_small.csv", index=False)

print("\nSeparate files created for big and small prawns")
print(f"Big prawns: {len(df_big)} detections")
print(f"Small prawns: {len(df_small)} detections") 