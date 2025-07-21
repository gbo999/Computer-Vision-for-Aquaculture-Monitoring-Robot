import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import glob
import re

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../../..'))

# Data paths
combined_body_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/combined_body_length_data.csv'
combined_carapace_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/combined_carapace_length_data.csv'

# Output directory for visualizations
output_dir =r'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/center_verification_cropped'
os.makedirs(output_dir, exist_ok=True)

# Directory containing cropped images - update this path to point to your cropped images
cropped_image_dirs = [
    '/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace/crops',
    # Add more directories if needed
]

def find_cropped_image(image_name):
    """Find the cropped image file based on the image name in the dataframe."""
    if not isinstance(image_name, str):
        return None
    
    # Look for exact matches first
    for img_dir in cropped_image_dirs:
        # Try exact match
        exact_path = os.path.join(img_dir, image_name)
        if os.path.exists(exact_path):
            return exact_path
        
        # Try with common image extensions
        for ext in ['.jpg', '.png', '.jpeg']:
            if os.path.exists(exact_path + ext):
                return exact_path + ext
    
    # Try glob pattern matching if exact match fails
    for img_dir in cropped_image_dirs:
        pattern = os.path.join(img_dir, f"*{image_name}*")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    return None

def visualize_centers_on_cropped(combined_data_path, measurement_type='body'):
    """
    Create visualizations of center points on cropped images.
    
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
    
    # Check for required columns
    if 'meas_image_name' not in df.columns:
        print("Required column 'meas_image_name' not found in dataframe")
        return
    
    # Check for center coordinates
    center_x_col = 'meas_center_x' if 'meas_center_x' in df.columns else None
    center_y_col = 'meas_center_y' if 'meas_center_y' in df.columns else None
    
    if not center_x_col or not center_y_col:
        # Try alternative names
        if 'center_x' in df.columns:
            center_x_col = 'center_x'
            center_y_col = 'center_y'
        else:
            print("Could not find center coordinate columns")
            cols = [col for col in df.columns if 'center' in col.lower()]
            print(f"Available center-related columns: {cols}")
            return
    
    # Ensure match type column exists
    match_type_col = 'match_type' if 'match_type' in df.columns else None
    
    # Process a subset of images or all if fewer than the limit
    sample_size = min(30, len(df))
    samples = df.sample(sample_size) if len(df) > sample_size else df
    
    created_count = 0
    for idx, row in samples.iterrows():
        # Get image name
        image_name = row['meas_image_name']
        
        # Find the cropped image
        cropped_image_path = find_cropped_image(image_name)
        if not cropped_image_path:
            print(f"Could not find cropped image for {image_name}")
            # Try to save the path components for debugging
            if isinstance(image_name, str) and len(image_name) > 0:
                print(f"  Path components: {image_name.split('/')}")
            continue
        
        print(f"Processing cropped image: {os.path.basename(cropped_image_path)}")
        
        try:
            # Load the cropped image
            img = Image.open(cropped_image_path)
            img_width, img_height = img.size
            
            # Get center coordinates (normalized or absolute)
            center_x = row[center_x_col]
            center_y = row[center_y_col]
            
            # Check if coordinates need scaling
            # Typically, normalized coordinates are between 0 and 1
            if 0 <= center_x <= 1 and 0 <= center_y <= 1:
                # Coordinates are normalized, scale to image dimensions
                center_x = center_x * img_width
                center_y = center_y * img_height
            
            # Create visualization
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(np.array(img))
            
            # Draw center point
            ax.scatter(center_x, center_y, color='lime', s=100, marker='x', linewidth=2)
            ax.text(center_x+5, center_y+5, f"Center", color='lime', fontsize=10, fontweight='bold')
            
            # Add grid lines to help verify center position
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            
            # Add match type if available
            if match_type_col and match_type_col in row:
                match_type = row[match_type_col]
                ax.set_title(f"Match Type: {match_type}", fontsize=12)
            
            # Add image info
            image_info = f"Image: {os.path.basename(cropped_image_path)}\n"
            image_info += f"Dimensions: {img_width}x{img_height}"
            ax.text(10, 30, image_info, color='white', fontsize=10,
                   bbox=dict(facecolor='black', alpha=0.7))
            
            # Add ruler markings on axes
            ax.set_xticks(np.arange(0, img_width, img_width/10))
            ax.set_yticks(np.arange(0, img_height, img_height/10))
            
            # Save visualization
            output_filename = f"{measurement_type}_{idx}_cropped_verification.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            created_count += 1
            print(f"Saved visualization to {output_path}")
            
        except Exception as e:
            print(f"Error processing image {cropped_image_path}: {e}")
    
    print(f"Created {created_count} visualizations for {measurement_type} data")

def main():
    print("Starting center point verification on cropped images...")
    
    # Process body data
    if os.path.exists(combined_body_path):
        visualize_centers_on_cropped(combined_body_path, 'body')
    else:
        print(f"Body data file not found: {combined_body_path}")
    
    # Process carapace data
    if os.path.exists(combined_carapace_path):
        visualize_centers_on_cropped(combined_carapace_path, 'carapace')
    else:
        print(f"Carapace data file not found: {combined_carapace_path}")
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 