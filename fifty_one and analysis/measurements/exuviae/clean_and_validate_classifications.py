import fiftyone as fo
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load the dataset
dataset = fo.load_dataset("prawn_keypoints")
print(f"Loaded dataset with {len(dataset)} samples")

def validate_bbox(bbox, img_width, img_height):
    """Validate and fix bounding box coordinates"""
    if bbox is None:
        return None
        
    x, y, w, h = bbox
    
    # Check if any coordinates are NaN or invalid
    if np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h):
        return None
        
    # Check if box has zero or negative dimensions
    if w <= 0 or h <= 0:
        return None
        
    # Check if box is outside image bounds (normalized coordinates)
    if x < 0 or x > 1 or y < 0 or y > 1:
        return None
    if x + w > 1 or y + h > 1:
        return None
        
    return [x, y, w, h]

def plot_bbox(ax, bbox, color, label, alpha=0.2, linewidth=4):
    """Plot a single bounding box"""
    x, y, w, h = bbox
    rect = plt.Rectangle((x, y), w, h, 
                        fill=True, 
                        alpha=alpha,
                        color=color, 
                        linewidth=linewidth)
    ax.add_patch(rect)
    
    # Add label above box
    ax.text(x, y-0.02, label, 
            color=color,
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Process each sample
for sample in dataset:
    print(f"\nProcessing {sample.filepath}...")
    
    # Get image dimensions
    img = Image.open(sample.filepath)
    img_width, img_height = img.size
    
    # Track valid detections for this image
    valid_detections = []
    
    # Process manual classifications (if any)
    if hasattr(sample, "detections") and sample.detections is not None:
        for det in sample.detections.detections:
            bbox = det.bounding_box
            valid_bbox = validate_bbox(bbox, img_width, img_height)
            
            if valid_bbox is not None:
                if hasattr(det, "tags") and det.tags:
                    # Get size from tags
                    if 'big' in det.tags:
                        det.label = 'big'
                    elif 'small' in det.tags:
                        det.label = 'small'
                    valid_detections.append(det)
    
    # Process Shai's measurements (if any)
    if hasattr(sample, "bounding_box") and sample.bounding_box is not None:
        for det in sample.bounding_box.detections:
            bbox = det.bounding_box
            valid_bbox = validate_bbox(bbox, img_width, img_height)
            
            if valid_bbox is not None:
                # Convert Shai's length to category
                length = float(det.label)
                if length > 160:  # Threshold for big/small classification
                    det.label = 'big_shai'
                else:
                    det.label = 'small_shai'
                valid_detections.append(det)
    
    # Update sample with valid detections only
    sample.detections = fo.Detections(detections=valid_detections)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    
    # Plot all valid detections
    for det in valid_detections:
        bbox = det.bounding_box
        if 'shai' in det.label:
            # Shai's measurements
            color = 'purple' if det.label == 'big_shai' else 'cyan'
        else:
            # Manual classifications
            color = 'red' if det.label == 'big' else 'blue'
        
        # Convert normalized coordinates to pixel coordinates for plotting
        plot_bbox(plt.gca(), 
                 [bbox[0]*img_width, bbox[1]*img_height, 
                  bbox[2]*img_width, bbox[3]*img_height],
                 color, det.label)
    
    plt.title(f"Classifications for {os.path.basename(sample.filepath)}\n"
              f"Red=Big, Blue=Small, Purple=Big(Shai), Cyan=Small(Shai)")
    plt.axis('off')
    
    # Save visualization
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"cleaned_classification_{os.path.basename(sample.filepath)}"),
                bbox_inches='tight', dpi=300)
    plt.close()

# Save the cleaned dataset
dataset.save()

# Print statistics
print("\nClassification Statistics:")
print("-" * 50)

total_big = 0
total_small = 0
total_big_shai = 0
total_small_shai = 0

for sample in dataset:
    if hasattr(sample, "detections") and sample.detections is not None:
        for det in sample.detections.detections:
            if det.label == 'big':
                total_big += 1
            elif det.label == 'small':
                total_small += 1
            elif det.label == 'big_shai':
                total_big_shai += 1
            elif det.label == 'small_shai':
                total_small_shai += 1

print("\nManual Classifications:")
print(f"Big: {total_big}")
print(f"Small: {total_small}")
print("\nShai's Measurements:")
print(f"Big: {total_big_shai}")
print(f"Small: {total_small_shai}") 