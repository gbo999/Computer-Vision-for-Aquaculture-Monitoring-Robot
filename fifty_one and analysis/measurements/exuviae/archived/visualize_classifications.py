import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import random

# Load the exported data
manual_bbox_path = "spreadsheet_files/manual_classifications_updated_20250713_113720.csv"
images_dir = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized"

df_manual = pd.read_csv(manual_bbox_path)

def validate_bbox(bbox, img_width, img_height):
    """Validate and fix bounding box coordinates if needed"""
    x, y, w, h = bbox
    
    # Check if any coordinates are NaN
    if np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h):
        return None
        
    # Check if box has zero or negative dimensions
    if w <= 0 or h <= 0:
        return None
        
    # Convert normalized coordinates (0-1) to pixel coordinates
    x = x * img_width
    y = y * img_height
    w = w * img_width
    h = h * img_height
    
    return [x, y, w, h]

def plot_bbox(ax, bbox, color, label, img_width, img_height):
    """Plot a bounding box on the axes"""
    # Validate and convert coordinates
    pixel_bbox = validate_bbox(bbox, img_width, img_height)
    if pixel_bbox is None:
        print(f"Invalid bbox: {bbox}")
        return
        
    x, y, w, h = pixel_bbox
    
    # Draw filled rectangle with transparency
    rect_fill = plt.Rectangle((x, y), w, h, fill=True, color=color, alpha=0.2)
    ax.add_patch(rect_fill)
    
    # Draw border with thicker line
    rect_border = plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=4)
    ax.add_patch(rect_border)
    
    # Add label text above the box with larger font and background
    plt.text(x, y-20, label, color=color, fontsize=14, weight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, linewidth=2))

# Create figure for statistics
plt.figure(figsize=(15, 5))

# Plot overall distribution
plt.subplot(121)
sizes = df_manual['manual_size'].value_counts()
plt.bar(sizes.index, sizes.values, color=['red', 'blue'])
plt.title('Distribution of Manual Classifications')
plt.ylabel('Count')

# Plot size distribution by image
plt.subplot(122)
size_by_image = df_manual.groupby(['image_name', 'manual_size']).size().unstack(fill_value=0)
size_by_image.plot(kind='bar', stacked=True)
plt.title('Classifications by Image')
plt.xticks(rotation=90)
plt.tight_layout()

# Save statistics plot
save_dir = "visualizations"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "classification_statistics.png"), bbox_inches='tight')
plt.close()

# Get all images that have manual classifications (not untagged)
classified_df = df_manual[df_manual['manual_size'] != 'untagged']
sample_images = classified_df['image_name'].unique()
print(f"Visualizing {len(sample_images)} images with manual classifications")

invalid_boxes = []
for img_name in sample_images:
    # Clean up image name and construct path
    base_name = img_name.replace('colored_', '')
    if not base_name.endswith('.jpg'):
        base_name += '.jpg'
    img_path = os.path.join(images_dir, base_name)
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
        
    img = Image.open(img_path)
    img_width, img_height = img.size
    
    # Create figure
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    
    # Plot manual classifications (only show classified ones, not untagged)
    manual_boxes = df_manual[(df_manual['image_name'] == img_name) & (df_manual['manual_size'] != 'untagged')]
    for _, row in manual_boxes.iterrows():
        bbox = [row['bbox_x'], row['bbox_y'], row['bbox_width'], row['bbox_height']]
        # Check if any coordinates are invalid
        if any(np.isnan(coord) for coord in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
            invalid_boxes.append({
                'image': img_name,
                'size': row['manual_size'],
                'bbox': bbox
            })
            continue
            
        color = 'red' if row['manual_size'] == 'big' else 'blue'
        label = f"{row['manual_size']}"
        plot_bbox(plt.gca(), bbox, color, label, img_width, img_height)
    
    plt.title(f"Image: {base_name}\nRed=Big, Blue=Small")
    plt.axis('off')
    
    # Save the visualization
    plt.savefig(os.path.join(save_dir, f"classification_{base_name}"), 
                bbox_inches='tight', dpi=300)
    plt.close()

print("\nClassification Statistics:")
print("-" * 50)
print("\nTotal counts:")
print(df_manual['manual_size'].value_counts())

print("\nCounts per image (classified only):")
classified_counts = classified_df.groupby(['image_name', 'manual_size']).size().unstack(fill_value=0)
print(classified_counts)

if invalid_boxes:
    print("\nWARNING: Found invalid bounding boxes:")
    for box in invalid_boxes:
        print(f"Image: {box['image']}, Size: {box['size']}, Box: {box['bbox']}")

print("\nVisualizations saved in 'visualizations' directory")
print("Red boxes = Big prawns")
print("Blue boxes = Small prawns") 