import cv2
import numpy as np
import matplotlib.pyplot as plt
from chessboard_harris import improved_chessboard_detection, detect_chessboard_harris

# Replace this with the actual path to your chessboard image
image_path = '/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/1/carapace/right/undistorted_GX010068_27_795.jpg_gamma.jpg'

# First try with OpenCV's standard detector with different pattern sizes
pattern_sizes = [(9,6), (8,5), (7,6), (7,7)]
square_size = None

for pattern in pattern_sizes:
    print(f"\nTrying pattern size: {pattern}")
    result = improved_chessboard_detection(image_path, pattern_size=pattern)
    if result is not None and result > 5:  # Sanity check: square size should be reasonable
        square_size = result
        print(f"Success with pattern size {pattern}!")
        break

# If standard detection failed, try the more specialized Harris detector directly with stricter params
if square_size is None or square_size < 5:
    print("\nTrying specialized Harris detector with strict parameters...")
    
    # Parameters to try (more strict filtering):
    params = [
        {'block_size': 3, 'ksize': 5, 'threshold_ratio': 0.07, 'min_distance': 20},
        {'block_size': 4, 'ksize': 7, 'threshold_ratio': 0.05, 'min_distance': 25},
        {'block_size': 2, 'ksize': 3, 'threshold_ratio': 0.1, 'min_distance': 15}
    ]
    
    for i, p in enumerate(params):
        print(f"\nTrying Harris parameter set {i+1}:")
        print(p)
        
        result = detect_chessboard_harris(
            image_path, 
            block_size=p['block_size'], 
            ksize=p['ksize'], 
            threshold_ratio=p['threshold_ratio'],
            min_distance=p['min_distance']
        )
        
        if result is not None and result > 5:
            square_size = result
            print(f"Success with parameter set {i+1}!")
            break

if square_size is not None and square_size > 5:
    print(f"\nFinal chessboard square size: {square_size:.2f} pixels")
    
    # If you know the real-world size of the chessboard squares (e.g., in mm),
    # you can calculate the conversion factor:
    REAL_SQUARE_SIZE_MM = 20  # Replace with actual size in mm
    px_to_mm_ratio = REAL_SQUARE_SIZE_MM / square_size
    print(f"Conversion factor: {px_to_mm_ratio:.4f} mm/pixel")
else:
    print("\nFailed to detect chessboard with reliable parameters.")
    print("Try improving image quality, adjusting lighting, or using a different image.") 