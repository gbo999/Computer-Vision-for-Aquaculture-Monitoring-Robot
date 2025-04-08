import cv2
import numpy as np
import matplotlib.pyplot as plt
from chessboard_grid_detector import detect_chessboard_grid, main as detect_main

print("=" * 60)
print("CHESSBOARD GRID DETECTION USING HOUGH LINES METHOD")
print("=" * 60)
print("This approach directly detects the chessboard grid using Hough Lines")
print("Based on: https://medium.com/@siromermer/extracting-chess-square-coordinates-76b933f0f64e")
print("=" * 60)

# Replace this with the actual path to your chessboard image
image_path = '/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/1/carapace/right/undistorted_GX010068_27_795.jpg_gamma.jpg'

# Run the detection
square_size, grid_info = detect_chessboard_grid(image_path)

if square_size is not None:
    # Output the square size and conversion factor
    print("\nSUCCESS! Detected chessboard grid.")
    print(f"Square size: {square_size:.2f} pixels")
    
    # If you know the real-world size of the chessboard squares (in mm),
    # calculate the conversion factor
    REAL_SQUARE_SIZE_MM = 20  # Replace with actual size in mm
    px_to_mm_ratio = REAL_SQUARE_SIZE_MM / square_size
    
    print(f"\nConversion factor:")
    print(f"  {px_to_mm_ratio:.4f} mm/pixel")
    print(f"  {1/px_to_mm_ratio:.4f} mm per pixel")
    
    # Additional grid information
    if grid_info:
        print(f"\nGrid details:")
        print(f"  Grid points: {len(grid_info['grid_points'])}")
        print(f"  Grid lines: {len(grid_info['h_lines'])} horizontal, {len(grid_info['v_lines'])} vertical")
        
        # Number of squares in the grid
        rows_count = len(grid_info['rows'])
        cols_count = max(len(row) for row in grid_info['rows'])
        print(f"  Approximate grid size: {rows_count-1}×{cols_count-1} squares")
else:
    print("\nFAILED to detect chessboard grid.")
    print("Try with a different image or method.")
    
    # Try another approach as fallback
    print("\nTrying alternative approaches:")
    
    # Try FAST corner detector
    try:
        from fast_corner_detector import try_multiple_thresholds
        print("\n1. Trying FAST corner detector...")
        
        result = try_multiple_thresholds(
            image_path, 
            thresholds=[40, 30, 20, 15],
            grid_strict=False  # More permissive
        )
        
        if result is not None and result > 5:
            print(f"Success with FAST detector! Square size: {result:.2f} pixels")
        else:
            print("FAST detector failed.")
    except ImportError:
        print("FAST detector not available.")
    
    # Try standard OpenCV detector as last resort
    try:
        from chessboard_harris import improved_chessboard_detection
        print("\n2. Trying standard OpenCV detector...")
        
        for pattern in [(9,6), (8,5), (7,7), (6,6)]:
            print(f"Trying pattern {pattern}...")
            result = improved_chessboard_detection(image_path, pattern_size=pattern)
            if result is not None and result > 5:
                print(f"Success with pattern {pattern}! Square size: {result:.2f} pixels")
                break
        else:
            print("OpenCV detector failed with all patterns.")
    except ImportError:
        print("OpenCV detector not available.") 