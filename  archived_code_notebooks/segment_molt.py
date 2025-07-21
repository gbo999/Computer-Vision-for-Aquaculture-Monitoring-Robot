import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm



def segment_molt(image_path, output_path):
    img = cv2.imread(image_path)
    if img is not None:
        # Define colors (BGR format)
        TURQUOISE_COLOR = np.array([31, 156, 212])  # Brownish color for exuviae
        BROWNISH_COLOR = np.array([79, 66, 52])  # Turquoise for background

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple threshold for dark areas (black backgrounds)
        _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)  # Note: Removed _INV
        
        # Create the segmented image
        segmented = np.zeros_like(img)
        
        # Apply colors directly - everything turquoise except dark areas
        segmented[mask > 0] = BROWNISH_COLOR  # Light areas get turquoise
        segmented[mask == 0] = TURQUOISE_COLOR
        
        # Save the result and convert to RGB
        cv2.imwrite(output_path, cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB) )


def process_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(input_folder).glob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image with progress bar
    for img_path in tqdm(image_files):
        try:
            # Get output path
            output_path = Path(output_folder) / f"segmented_{img_path.name}"
            
            # Process image
            segment_molt(str(img_path), str(output_path))
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
    
    print("Processing complete!")

if __name__ == "__main__":
    # Define input and output folders
    input_folder = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized"
    output_folder = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized/segmented"
    
    # Process all images
    process_folder(input_folder, output_folder) 