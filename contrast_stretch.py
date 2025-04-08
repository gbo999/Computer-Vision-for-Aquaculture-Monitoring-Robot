import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def contrast_stretch(image, lower_percentile=0.5, upper_percentile=99.5, gamma=0.7):
    """
    Apply contrast stretching to an image with more aggressive parameters and gamma correction.
    
    Args:
        image: Input image (numpy array)
        lower_percentile: Lower percentile for stretching (default: 0.5)
        upper_percentile: Upper percentile for stretching (default: 99.5)
        gamma: Gamma correction value (default: 0.7, lower values increase contrast)
    
    Returns:
        Contrast stretched image
    """
    # Convert to float32 for calculations
    img_float = image.astype(np.float32)
    
    # Calculate percentiles
    lower = np.percentile(img_float, lower_percentile)
    upper = np.percentile(img_float, upper_percentile)
    
    # Apply contrast stretching
    stretched = (img_float - lower) * (255.0 / (upper - lower))
    stretched = np.clip(stretched, 0, 255)
    
    # Apply gamma correction
    stretched = np.power(stretched / 255.0, gamma) * 255.0
    
    return stretched.astype(np.uint8)

def process_directory(input_dir, output_dir):
    """
    Process all images in a directory with contrast stretching.
    
    Args:
        input_dir: Path to input directory
        output_dir: Path to output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
    
    # Process each image
    for filename in tqdm(image_files, desc="Processing images"):
        # Read image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read image: {filename}")
            continue
        
        # Apply contrast stretching to each channel
        if len(img.shape) == 3:  # Color image
            stretched = np.zeros_like(img)
            for i in range(3):
                stretched[:,:,i] = contrast_stretch(img[:,:,i])
        else:  # Grayscale image
            stretched = contrast_stretch(img)
        
        # Save the result
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, stretched)

if __name__ == "__main__":
    # Input and output directories
    input_dir = "/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/all"
    output_dir = os.path.join(os.path.dirname(input_dir), "contrast_stretched")
    
    # Process the directory
    process_directory(input_dir, output_dir)
    print(f"Processing complete. Stretched images saved to: {output_dir}") 