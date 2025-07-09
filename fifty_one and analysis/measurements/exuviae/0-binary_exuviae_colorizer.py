"""
Binary Molt Colorizer

This script processes images of white prawn molts (exuviae) by applying realistic coloring
to make them look like actual prawns. It uses a simple but effective approach that 
identifies the white molt against the dark background using binary thresholding, then 
applies natural prawn coloring for better visualization.

Key Features:
- Binary thresholding to identify white molts
- Applies realistic prawn coloring (brown) to the molt
- Turquoise background for contrast
- Batch processing support for entire folders
- Progress tracking with tqdm
- Extensive error handling

Example Usage:
    python binary_molt_colorizer.py

The script will process all images in the input folder and save the colorized
versions in the output folder with "segmented_" prefix.

Author: Gil Benor
Date: March 4, 2024
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


def colorize_molt(image_path: str, output_path: str) -> bool:
    """
    Process a single molt image by identifying the white molt and applying natural prawn coloring.
    
    The function uses a straightforward approach:
    1. Convert image to grayscale
    2. Apply binary threshold to separate white molt from dark background
    3. Apply realistic prawn coloring to the molt and contrasting color to background
    
    Args:
        image_path (str): Path to the input image file
        output_path (str): Path where the processed image will be saved
        
    Returns:
        bool: True if processing was successful, False otherwise
        
    Note:
        The function uses fixed colors:
        - Molt areas (originally white): Brown [79, 66, 52] in BGR to match real prawn color
        - Background (originally dark): Turquoise [31, 156, 212] for contrast
        
        The threshold value of 60 was determined empirically to effectively separate
        white molts from the dark background in our imaging conditions.
    """
    img = cv2.imread(image_path)
    if img is not None:
        # Define colors (BGR format)
        PRAWN_COLOR = np.array([79, 66, 52])      # Brown color to match real prawns
        BG_COLOR = np.array([31, 156, 212])       # Turquoise for background contrast

        # Convert to grayscale for thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Binary threshold (molt appears white against dark background)
        _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        
        # Create output image and apply colors
        colorized = np.zeros_like(img)
        colorized[mask > 0] = PRAWN_COLOR     # White areas (molt) become brown
        colorized[mask == 0] = BG_COLOR       # Dark areas become turquoise
        
        # Save result in RGB format
        cv2.imwrite(output_path, cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
        return True
    return False


def process_folder(input_folder: str, output_folder: str) -> None:
    """
    Process all images in a folder, applying molt colorization to each.
    
    Args:
        input_folder (str): Path to folder containing input images
        output_folder (str): Path where processed images will be saved
        
    The function will:
    1. Create the output folder if it doesn't exist
    2. Find all image files with supported extensions
    3. Process each image and save the result with "segmented_" prefix
    4. Show progress with tqdm
    5. Handle and report any errors during processing
    """
    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    
    # Find all image files (case insensitive)
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(input_folder).glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image with progress tracking
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            output_path = Path(output_folder) / f"segmented_{img_path.name}"
            if colorize_molt(str(img_path), str(output_path)):
                tqdm.write(f"Successfully processed {img_path.name}")
            else:
                tqdm.write(f"Failed to read {img_path.name}")
        except Exception as e:
            tqdm.write(f"Error processing {img_path.name}: {str(e)}")
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    # Default paths - modify these or add command line arguments as needed
    input_folder = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized"
    output_folder = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized/segmented"
    
    # Process all images
    process_folder(input_folder, output_folder) 