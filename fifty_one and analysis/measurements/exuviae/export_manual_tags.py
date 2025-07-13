import fiftyone as fo
import pandas as pd
import numpy as np
import os

# Load the dataset
dataset = fo.load_dataset("prawn_keypoints")
print(f"Loaded dataset with {len(dataset)} samples")

# Prepare data for export
export_data = []

for sample in dataset:
    if sample.detections is None:
        continue
        
    for detection in sample.detections.detections:
        # Get size from tags (if any)
        size = None
        if hasattr(detection, 'tags') and detection.tags:
            if 'big' in detection.tags:
                size = 'big'
            elif 'small' in detection.tags:
                size = 'small'
        
        # If no tags, check the label
        if size is None:
            if detection.label in ['big', 'small']:
                size = detection.label
            else:
                size = 'untagged'  # Include untagged detections
                
        # Get bounding box coordinates
        bbox = detection.bounding_box  # [x, y, width, height] in normalized coordinates
        if bbox is None:
            continue
            
        # Calculate real-world coordinates (mm)
        img_width_mm = 5312
        img_height_mm = 2988
        
        # Create data entry
        entry = {
            'image_name': sample.filename,
            'manual_size': size,
            # Normalized coordinates (0-1)
            'bbox_x': bbox[0],
            'bbox_y': bbox[1],
            'bbox_width': bbox[2],
            'bbox_height': bbox[3],
            # Real-world coordinates (mm)
            'bbox_x_mm': bbox[0] * img_width_mm,
            'bbox_y_mm': bbox[1] * img_height_mm,
            'bbox_width_mm': bbox[2] * img_width_mm,
            'bbox_height_mm': bbox[3] * img_height_mm,
        }
        
        export_data.append(entry)

# Convert to DataFrame and export
df = pd.DataFrame(export_data)

# Create export directory if it doesn't exist
export_dir = "spreadsheet_files"
os.makedirs(export_dir, exist_ok=True)

# Export to CSV with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
export_path = os.path.join(export_dir, f"manual_classifications_updated_{timestamp}.csv")
df.to_csv(export_path, index=False)

print(f"\nExported {len(export_data)} detections to {export_path}")
print("\nExported data includes:")
print("- Image name")
print("- Manual size classification (big/small/untagged)")
print("- Bounding box coordinates (normalized and in mm)")

# Print statistics
print(f"\nClassification counts:")
print(f"- Big: {len(df[df['manual_size'] == 'big'])}")
print(f"- Small: {len(df[df['manual_size'] == 'small'])}")
print(f"- Untagged: {len(df[df['manual_size'] == 'untagged'])}")

# Show duplicates
duplicates = df.groupby(['image_name', 'manual_size']).size().reset_index(name='count')
duplicates = duplicates[duplicates['count'] > 1]
if len(duplicates) > 0:
    print(f"\nFound {len(duplicates)} duplicate classifications:")
    for _, row in duplicates.iterrows():
        print(f"- {row['image_name']}: {row['count']} {row['manual_size']} detections")
else:
    print("\nNo duplicate classifications found!") 