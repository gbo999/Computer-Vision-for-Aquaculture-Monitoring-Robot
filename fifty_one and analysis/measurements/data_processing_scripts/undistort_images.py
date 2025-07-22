"""
GoPro Image Undistortion Script

This script undistorts images captured by a GoPro Hero 11 camera using camera parameters obtained
from Gyroflow camera calibration. It processes all JPG images in an input directory and saves 
the undistorted versions to an output directory.

Camera Specifications:
    - Model: GoPro Hero 11
    - Resolution: 5312x2988 pixels
    - Camera Matrix: Obtained from Gyroflow calibration
        - Focal Length: ~3043px (horizontal), ~3015px (vertical)
        - Principal Point: (2525, 1596)
    - Distortion Coefficients: [0.253, 0.132, 0.070, 0.161]
        - Obtained from Gyroflow's fisheye lens calibration

The script uses OpenCV's fisheye camera model for undistortion since GoPro cameras have significant
wide-angle distortion. The camera matrix and distortion coefficients were exported from Gyroflow,
which performs camera calibration using rolling shutter correction and lens distortion analysis.

Usage:
    1. Set input_dir to the directory containing distorted images
    2. Set output_dir where undistorted images will be saved
    3. Run the script: python undistort_images.py

Dependencies:
    - OpenCV (cv2)
    - NumPy
    - tqdm (for progress bar)
"""

import numpy as np
import cv2
import os
from tqdm import tqdm

# Camera calibration parameters for GoPro Hero 11
CAMERA_MATRIX = np.array([
    [3043.621958852673, 0.0, 2525.2907991218367],
    [0.0, 3015.014085286132, 1596.4745000545388],
    [0.0, 0.0, 1.0]
])

DISTORTION_COEFFS = np.array([
    0.2528090891288297,
    0.1324221379663344,
    0.07048721428221141,
    0.1610412047411928
])

def undistort_image(img, camera_matrix, dist_coeffs, balance=1.0, fov_scale=0.98):
    """
    Undistort a single image using fisheye camera model.
    
    Args:
        img (np.ndarray): Input image to undistort
        camera_matrix (np.ndarray): 3x3 camera intrinsic matrix
        dist_coeffs (np.ndarray): Distortion coefficients [k1, k2, k3, k4]
        balance (float): Balance value between 0-1 for undistortion strength
        fov_scale (float): Scale factor for field of view after undistortion
    
    Returns:
        np.ndarray: Undistorted image
    """
    h, w = img.shape[:2]
    
    # Calculate new camera matrix for undistortion
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, dist_coeffs, (w, h), np.eye(3), 
        balance=balance, fov_scale=fov_scale
    )
    
    # Generate undistortion maps
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), new_K, 
        (w, h), cv2.CV_16SC2
    )
    
    # Apply undistortion
    return cv2.remap(img, map1, map2, 
                    interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_CONSTANT)

def process_directory(input_dir, output_dir):
    """
    Process all JPG images in input directory and save undistorted versions.
    
    Args:
        input_dir (str): Path to directory containing distorted images
        output_dir (str): Path where undistorted images will be saved
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each jpg file
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(".jpg") and not filename.startswith("undistorted"):
            # Load image
            image_path = os.path.join(input_dir, filename)
            img = cv2.imread(image_path)
            
            if img is None:
                print(f"Error: Image {filename} not loaded")
                continue
            
            # Undistort image
            undistorted_image = undistort_image(img, CAMERA_MATRIX, DISTORTION_COEFFS)
            
            # Save result
            output_path = os.path.join(output_dir, f"undistorted_{filename}")
            cv2.imwrite(output_path, undistorted_image)

if __name__ == "__main__":
    # Configuration
    input_dir = "path/to/your/input/directory"
    output_dir = "path/to/your/output_undistorted/directory"
    # Run processing
    process_directory(input_dir, output_dir)
    print("Processing completed.")