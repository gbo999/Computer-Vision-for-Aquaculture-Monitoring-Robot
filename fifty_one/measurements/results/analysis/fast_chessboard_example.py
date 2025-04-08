import cv2
import numpy as np
import matplotlib.pyplot as plt
from fast_corner_detector import detect_chessboard_corners_fast, try_multiple_thresholds

print("=" * 60)
print("CHESSBOARD DETECTION USING LIGHTWEIGHT FAST CORNER DETECTOR")
print("=" * 60)

# Replace this with the actual path to your chessboard image
image_path = '/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/1/carapace/right/undistorted_GX010068_27_795.jpg_gamma.jpg'

# APPROACH 1: Try with progressively lower thresholds (more permissive)
thresholds = [60, 50, 40, 30, 20, 15, 10]

print("\nAPPROACH 1: Trying multiple thresholds with grid enforcement")
print("-" * 50)
square_size = try_multiple_thresholds(
    image_path, 
    thresholds=thresholds,
    nonmax_suppression=True,  # Use non-maximum suppression for cleaner corners
    grid_strict=True          # Enforce strict grid structure
)

# If the first approach failed, try again without grid enforcement
if square_size is None:
    print("\nAPPROACH 2: Trying without strict grid enforcement")
    print("-" * 50)
    
    # Try without grid enforcement (might be useful for partial chessboards)
    square_size = try_multiple_thresholds(
        image_path, 
        thresholds=[40, 30, 20, 15, 10],  # Start with medium thresholds
        nonmax_suppression=True,
        grid_strict=False  # Don't require a perfect grid structure
    )

# Final output
if square_size is not None and square_size > 5:
    print("\n" + "=" * 50)
    print(f"FINAL RESULT: Chessboard square size: {square_size:.2f} pixels")
    
    # If you know the real-world size of the chessboard squares (e.g., in mm),
    # you can calculate the conversion factor:
    REAL_SQUARE_SIZE_MM = 20  # Replace with actual size in mm
    px_to_mm_ratio = REAL_SQUARE_SIZE_MM / square_size
    print(f"Conversion factor: {px_to_mm_ratio:.4f} mm/pixel")
    print(f"1 pixel = {1/px_to_mm_ratio:.4f} mm")
    print("=" * 50)
else:
    print("\n" + "=" * 50)
    print("FAILED to detect a valid chessboard pattern.")
    print("Suggestions:")
    print("1. Try preprocessing the image to improve contrast")
    print("2. Ensure the chessboard is clearly visible in the image")
    print("3. Try a different image if possible")
    print("=" * 50) 