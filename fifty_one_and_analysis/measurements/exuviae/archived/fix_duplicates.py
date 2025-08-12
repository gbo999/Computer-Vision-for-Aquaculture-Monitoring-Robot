import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Load the CSV file
csv_path = "spreadsheet_files/manual_classifications_with_bboxes.csv"
df = pd.read_csv(csv_path)

# Directory containing the images
images_dir = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized"

def plot_bbox(ax, bbox, color, label, alpha=0.2, linewidth=4):
    """Plot a single bounding box"""
    x, y, w, h = bbox
    rect = plt.Rectangle((x, y), w, h, 
                        fill=True, 
                        alpha=alpha,
                        color=color, 
                        linewidth=linewidth)
    ax.add_patch(rect)
    
    # Add label with coordinates
    ax.text(x, y-0.02, f"{label}\n({x:.3f}, {y:.3f}, {w:.3f}, {h:.3f})", 
            color=color,
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Find duplicates
duplicates = df.groupby(['image_name', 'manual_size']).size().reset_index(name='count')
duplicates = duplicates[duplicates['count'] > 1]

# Create output directory
output_dir = "fixed_classifications"
os.makedirs(output_dir, exist_ok=True)

# Create new DataFrame for fixed classifications
fixed_df = pd.DataFrame(columns=df.columns)

# Process each duplicate case
for _, row in duplicates.iterrows():
    image_name = row['image_name']
    size = row['manual_size']
    
    # Get entries for this image/size combination
    entries = df[(df['image_name'] == image_name) & 
                (df['manual_size'] == size)]
    
    # Load and display image
    base_name = image_name.replace('colored_', '')
    img_path = os.path.join(images_dir, base_name)
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
        
    img = Image.open(img_path)
    img_width, img_height = img.size
    
    # Create figure
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    
    # Plot bounding boxes
    for idx, entry in entries.iterrows():
        bbox = [entry['bbox_x'], entry['bbox_y'], 
                entry['bbox_width'], entry['bbox_height']]
        
        # Convert normalized coordinates to pixel coordinates
        bbox_pixels = [coord * (img_width if i % 2 == 0 else img_height) 
                      for i, coord in enumerate(bbox)]
        
        # Plot with different colors
        color = 'red' if idx == entries.index[0] else 'blue'
        label = f"Box {idx + 1}"
        plot_bbox(plt.gca(), bbox_pixels, color, label)
    
    plt.title(f"Image: {base_name}\nRed=Box 1, Blue=Box 2\nSize: {size}")
    plt.axis('off')
    
    # Save visualization
    plt.savefig(os.path.join(output_dir, f"duplicate_{base_name}"),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print information and get user input
    print(f"\nImage: {base_name}")
    print(f"Size: {size}")
    print("\nBounding boxes:")
    for idx, entry in entries.iterrows():
        print(f"\nBox {idx + 1}:")
        print(f"x: {entry['bbox_x']:.3f}")
        print(f"y: {entry['bbox_y']:.3f}")
        print(f"width: {entry['bbox_width']:.3f}")
        print(f"height: {entry['bbox_height']:.3f}")
        print(f"Area: {entry['bbox_width'] * entry['bbox_height']:.3f}")
    
    while True:
        choice = input("\nWhich box to keep? (1 or 2, or 's' to skip): ")
        if choice in ['1', '2', 's']:
            break
        print("Invalid choice. Please enter 1, 2, or s.")
    
    if choice == 's':
        print("Skipping this image...")
        continue
    
    # Add chosen box to fixed DataFrame
    chosen_idx = entries.index[int(choice) - 1]
    fixed_df = pd.concat([fixed_df, entries.loc[[chosen_idx]]], ignore_index=True)
    
    print(f"Kept Box {choice} for {base_name}")

# Add non-duplicate entries to fixed DataFrame
non_duplicates = df.merge(duplicates, on=['image_name', 'manual_size'], how='left')
non_duplicates = non_duplicates[non_duplicates['count'].isna()]
fixed_df = pd.concat([fixed_df, non_duplicates.drop('count', axis=1)], ignore_index=True)

# Save fixed classifications
output_path = os.path.join(output_dir, "fixed_classifications.csv")
fixed_df.to_csv(output_path, index=False)
print(f"\nSaved fixed classifications to {output_path}")
print(f"Visualizations saved in {output_dir} directory") 