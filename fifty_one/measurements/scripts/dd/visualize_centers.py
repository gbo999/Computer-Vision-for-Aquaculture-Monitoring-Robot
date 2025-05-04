import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import glob
import re
import random
from matplotlib.colors import to_rgba

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../../..'))

# Data paths
combined_body_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/combined_body_length_data.csv'
combined_carapace_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/combined_carapace_length_data.csv'

# Output directory for visualizations
output_dir =r'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/center_verification'
os.makedirs(output_dir, exist_ok=True)

# Image source directories - will need to be updated with actual paths
original_image_dirs = [
    '/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/2/carapace',
    # Add more directories if images are stored in multiple locations
]

def find_original_image(image_name):
    """Find the original image file based on the camera frame ID."""
    # Extract GX frame ID
    camera_frame = None
    
    if '.jpg_gamma' in image_name:
        parts = image_name.split('.jpg_gamma')
        camera_frame = parts[0]
    else:
        parts = image_name.split('_obj')
        if len(parts) > 1:
            camera_frame = parts[0]
    
    if not camera_frame:
        return None
    
    # Look for the image in all possible directories
    for img_dir in original_image_dirs:
        # Try different patterns
        patterns = [
            f"{camera_frame}.jpg",
            f"{camera_frame}.png",
            f"*{camera_frame}*.jpg",
            f"*{camera_frame}*.png"
        ]
        
        for pattern in patterns:
            matches = glob.glob(os.path.join(img_dir, pattern))
            if matches:
                return matches[0]
    
    return None

def visualize_centers(combined_data_path, measurement_type='body'):
    """
    Create visualizations of center points on original images with bounding boxes.
    
    Args:
        combined_data_path: Path to the combined data CSV
        measurement_type: 'body' or 'carapace'
    """
    print(f"Processing {measurement_type} data from: {combined_data_path}")
    
    # Load combined data
    try:
        df = pd.read_csv(combined_data_path)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Verify needed columns exist
    required_cols = ['meas_center_x', 'meas_center_y', 'meas_image_name', 
                     'length_Label', 'match_type']
    
    # Check if columns use different naming conventions
    if 'meas_center_x' not in df.columns and 'center_x' in df.columns:
        df['meas_center_x'] = df['center_x']
        df['meas_center_y'] = df['center_y']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}")
        # Find column names that might contain bounding box data
        bbox_cols = [col for col in df.columns if 'BoundingBox_1' in col.lower() or 'box' in col.lower()]
        print(f"Available box-related columns: {bbox_cols}")
        
        # Find column names that might contain center coordinates
        center_cols = [col for col in df.columns if 'center' in col.lower() or 'coord' in col.lower()]
        print(f"Available center-related columns: {center_cols}")
        return
    
    # Find bounding box columns
    bbox_cols = [col for col in df.columns if ('BoundingBox' in col or 'box' in col.lower()) and not col.startswith('meas_')]
    if not bbox_cols:
        print("No bounding box columns found, looking for original bounding box columns...")
        # Check if full length dataframe columns are included
        length_bbox_cols = [col for col in df.columns 
                           if col.startswith('length_') and ('BoundingBox' in col or 'boundingbox' in col.lower())]
        if length_bbox_cols:
            print(f"Found length bounding box columns: {length_bbox_cols}")
            bbox_cols = length_bbox_cols
        else:
            print("No bounding box columns found in the dataframe.")
            return
    
    # Process a subset of images (change the count as needed)
    sample_size = min(20, len(df))
    samples = df.sample(sample_size) if len(df) > sample_size else df
    
    for idx, row in samples.iterrows():
        # Get image information
        image_name = row['meas_image_name']
        center_x = row['meas_center_x'] * 5312  # Scaling as in combine_lengths_improved.py
        center_y = row['meas_center_y'] * 2988
        match_type = row['match_type']
        
        # Find the original image
        original_image_path = find_original_image(row['meas_image_name'])
        if not original_image_path:
            print(f"Could not find original image for {row['meas_image_name']}")
            continue
        
        print(f"Processing image: {os.path.basename(original_image_path)}")
        
        try:
            # Load the original image
            img = Image.open(original_image_path)
            img_width, img_height = img.size
            
            # Create figure and axis
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(np.array(img))
            
            # Extract and draw bounding boxes
            for bbox_col in bbox_cols:
                if pd.isna(row[bbox_col]):
                    continue
                
                bbox_str = str(row[bbox_col])
                if '(' in bbox_str and ')' in bbox_str:
                    try:
                        # Extract values from format like (x, y, w, h)
                        bbox_str = bbox_str.strip('()').replace(' ', '')
                        bbox_values = [float(x.strip()) for x in bbox_str.split(',')]
                        if len(bbox_values) == 4:
                            bbox_x, bbox_y, bbox_w, bbox_h = bbox_values
                            
                            # Draw the rectangle
                            rect = patches.Rectangle(
                                (bbox_x, bbox_y), bbox_w, bbox_h, 
                                linewidth=2, edgecolor='r', facecolor='none'
                            )
                            ax.add_patch(rect)
                            
                            # Add text label for the bounding box
                            ax.text(bbox_x, bbox_y-5, f"Box {bbox_col}", color='r', fontsize=9)
                    except Exception as e:
                        print(f"Error parsing bounding box {bbox_str}: {e}")
            
            # Draw center point with a distinct color
            ax.scatter(center_x, center_y, color='lime', s=100, marker='x', linewidth=2)
            ax.text(center_x+10, center_y+10, f"Center ({center_x:.1f}, {center_y:.1f})", 
                   color='lime', fontsize=10, fontweight='bold')
            
            # Add match type information
            ax.set_title(f"Match Type: {match_type}", fontsize=12)
            
            # Add image info
            image_info = f"Image: {os.path.basename(original_image_path)}\n"
            ax.text(10, 30, image_info, color='white', fontsize=10, 
                   bbox=dict(facecolor='black', alpha=0.7))
            
            # Save visualization
            output_filename = f"{measurement_type}_{idx}_verification.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved visualization to {output_path}")
            
        except Exception as e:
            print(f"Error processing image {original_image_path}: {e}")
    
    print(f"Completed visualization for {measurement_type} data")

def main():
    print("Starting center point verification visualization...")
    
    # Process body data
    if os.path.exists(combined_body_path):
        visualize_centers(combined_body_path, 'body')
    else:
        print(f"Body data file not found: {combined_body_path}")
    
    # Process carapace data
    if os.path.exists(combined_carapace_path):
        visualize_centers(combined_carapace_path, 'carapace')
    else:
        print(f"Carapace data file not found: {combined_carapace_path}")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 