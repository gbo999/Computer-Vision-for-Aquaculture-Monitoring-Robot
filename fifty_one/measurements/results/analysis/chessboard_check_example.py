import cv2
import numpy as np
import matplotlib.pyplot as plt
from chessboard_harris import improved_chessboard_detection, detect_chessboard_harris

# Replace this with the actual path to your chessboard image
image_path = '/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/1/carapace/right/undistorted_GX010068_27_795.jpg_gamma.jpg'

# First try with OpenCV's standard detector with different pattern sizes
pattern_sizes = [(9,6), (8,6), (7,7), (6,6), (5,5)]
square_size = None

print("PHASE 1: Trying OpenCV's built-in chessboard detector with different pattern sizes")
print("-" * 80)

for pattern in pattern_sizes:
    print(f"\nTrying pattern size: {pattern}")
    result = improved_chessboard_detection(image_path, pattern_size=pattern)
    if result is not None and result > 5:  # Sanity check: square size should be reasonable
        square_size = result
        print(f"Success with pattern size {pattern}!")
        break

# If standard detection failed, try the more specialized Harris detector directly with stricter params
if square_size is None or square_size < 5:
    print("\nPHASE 2: Trying specialized Harris detector with strict grid enforcement")
    print("-" * 80)
    
    # Very strict parameters for precise grid detection
    params = [
        # Strict parameters with high threshold and noise reduction
        {'block_size': 3, 'ksize': 5, 'threshold_ratio': 0.2, 'min_distance': 25},
        
        # Even stricter threshold with different kernel
        {'block_size': 4, 'ksize': 7, 'threshold_ratio': 0.15, 'min_distance': 30},
        
        # Try with smaller block size but higher threshold
        {'block_size': 2, 'ksize': 3, 'threshold_ratio': 0.25, 'min_distance': 20}
    ]
    
    for i, p in enumerate(params):
        print(f"\nTrying strict Harris parameter set {i+1}:")
        print(p)
        
        result = detect_chessboard_harris(
            image_path, 
            block_size=p['block_size'], 
            ksize=p['ksize'], 
            threshold_ratio=p['threshold_ratio'],
            min_distance=p['min_distance'],
            max_corners=100  # Further limit corners for stricter detection
        )
        
        if result is not None and result > 5:
            square_size = result
            print(f"Success with parameter set {i+1}!")
            break

if square_size is not None and square_size > 5:
    print("\n" + "=" * 40)
    print(f"FINAL RESULT: Chessboard square size: {square_size:.2f} pixels")
    
    # If you know the real-world size of the chessboard squares (e.g., in mm),
    # you can calculate the conversion factor:
    REAL_SQUARE_SIZE_MM = 20  # Replace with actual size in mm
    px_to_mm_ratio = REAL_SQUARE_SIZE_MM / square_size
    print(f"Conversion factor: {px_to_mm_ratio:.4f} mm/pixel")
    print(f"This means 1 pixel = {1/px_to_mm_ratio:.4f} mm")
    print("=" * 40)
else:
    print("\n" + "=" * 40)
    print("FAILED to detect a valid chessboard pattern with any parameter set.")
    print("Suggestions:")
    print("1. Use an image with a clear, high-contrast chessboard")
    print("2. Ensure the chessboard is well-lit with minimal glare")
    print("3. Try preprocessing the image to improve contrast")
    print("4. If possible, try a different chessboard image")
    print("=" * 40) 