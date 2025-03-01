import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
def segment_molt(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image")

    # Define colors (BGR format)
    TURQUOISE_COLOR = np.array([31, 156, 212])  # Brownish color for exuviae
    AZURE_COLOR = np.array([79, 66, 52])  # Turquoise for background

    # Convert to HSV for better segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Create multiple masks for different characteristics of the exuviae
    # Mask for whitish/pale parts
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    
    # Mask for slightly darker/brownish parts
    lower_brown = np.array([15, 20, 100])
    upper_brown = np.array([30, 150, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combine masks
    mask = cv2.bitwise_or(mask_white, mask_brown)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Additional cleanup - remove small noise
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100  # Adjust this value based on your image
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    
    # Create the segmented image
    segmented = np.zeros_like(img)
    
    # Apply colors
    segmented[mask > 0] = AZURE_COLOR
    segmented[mask == 0] = TURQUOISE_COLOR
    
    # Save the result and convert to RGB
    cv2.imwrite(output_path, cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    return segmented

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
    
    # Process each image
    for img_path in tqdm(image_files) :
        try:
            # Get output path
            output_path = Path(output_folder) / f"segmented_{img_path.name}"
            
            # Process image
            print(f"Processing {img_path.name}...")
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