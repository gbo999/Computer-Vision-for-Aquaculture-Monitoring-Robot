import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from chessboard_segmentation_detector import detect_chessboard_with_segmentation

print("=" * 70)
print("CHESSBOARD DETECTION USING COLOR SEGMENTATION (StackOverflow Method)")
print("Based on: https://stackoverflow.com/questions/66225558/cv2-findchessboardcorners-fails-to-find-corners")
print("=" * 70)

# Path to the test image
image_path = '/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/1/carapace/right/undistorted_GX010068_27_795.jpg_gamma.jpg'

# Try a range of pattern sizes
pattern_sizes = [(7, 7), (9, 6), (8, 5), (6, 6), (8, 8), (5, 5), (4, 4), (4, 3), (3, 4), (3, 3), (2, 2), (2, 1), (1, 2), (1, 1)]
success = False

for pattern in pattern_sizes:
    print(f"\nAttempting detection with pattern size: {pattern}")
    square_size = detect_chessboard_with_segmentation(image_path, pattern_size=pattern)
    
    if square_size is not None:
        print(f"\nSUCCESS with pattern size {pattern}!")
        print(f"Square size: {square_size:.2f} pixels")
        
        # Calculate conversion factor
        REAL_SQUARE_SIZE_MM = 20  # Replace with actual size in mm
        px_to_mm_ratio = REAL_SQUARE_SIZE_MM / square_size
        
        print("\nMeasurement conversion factors:")
        print(f"  {px_to_mm_ratio:.4f} mm/pixel")
        print(f"  {1/px_to_mm_ratio:.4f} mm per pixel")
        
        success = True
        break

if not success:
    print("\n" + "=" * 70)
    print("DETECTION FAILED WITH ALL PATTERN SIZES")
    print("\nSuggestions:")
    print("1. Try adjusting the HSV thresholds in the detection function")
    print("2. Try preprocessing the image (adjust contrast, brightness)")
    print("3. Try using another image with clearer chessboard pattern")
    print("=" * 70)

    # Try one more time with custom HSV values
    print("\nAttempting with custom HSV thresholds...")
    
    # Function to try custom HSV range
    def detect_with_custom_hsv(image_path, hsv_lower, hsv_upper, pattern=(7, 7)):
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask with custom HSV range
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        # Apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
        dilated = cv2.dilate(mask, kernel, iterations=5)
        result = 255 - cv2.bitwise_and(dilated, mask)
        
        # Find chessboard corners
        result = np.uint8(result)
        ret, corners = cv2.findChessboardCornersSB(result, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE)
        
        if ret:
            print(f"Success with custom HSV range!")
            # Draw corners
            img_corners = img.copy()
            cv2.drawChessboardCorners(img_corners, pattern, corners, ret)
            
            # Display
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(result, cmap='gray')
            plt.title('Processed with Custom HSV')
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
            plt.title('Detected Corners')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            # Calculate square size
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined_corners = cv2.cornerSubPix(result, corners, (11, 11), (-1, -1), criteria)
            
            # Calculate distances
            h_distances = []
            v_distances = []
            
            for i in range(len(corners)):
                row = i // pattern[0]
                col = i % pattern[0]
                
                if col < pattern[0] - 1:
                    right_idx = row * pattern[0] + (col + 1)
                    dist = np.linalg.norm(corners[i] - corners[right_idx])
                    h_distances.append(dist)
                
                if row < pattern[1] - 1:
                    bottom_idx = (row + 1) * pattern[0] + col
                    dist = np.linalg.norm(corners[i] - corners[bottom_idx])
                    v_distances.append(dist)
            
            # Calculate square size
            square_size = (np.median(h_distances) + np.median(v_distances)) / 2
            print(f"Square size: {square_size:.2f} pixels")
            
            return square_size
        
        return None
    
    # Try with a few custom HSV ranges
    hsv_ranges = [
        # Lighter squares
        (np.array([0, 0, 150]), np.array([179, 50, 255])),
        # Darker squares
        (np.array([0, 0, 30]), np.array([179, 110, 150])),
        # Medium squares
        (np.array([0, 0, 80]), np.array([179, 80, 220]))
    ]
    
    for i, (lower, upper) in enumerate(hsv_ranges):
        print(f"\nTrying HSV range #{i+1}:")
        print(f"Lower: {lower}")
        print(f"Upper: {upper}")
        
        for pattern in [(7, 7), (9, 6), (6, 6), (8, 5), (5, 8), (8, 8), (5, 5), (4, 4), (4, 3), (3, 4), (3, 3), (2, 2), (2, 1), (1, 2), (1, 1)]:
            result = detect_with_custom_hsv(image_path, lower, upper, pattern)
            if result is not None:
                # Calculate conversion factor
                REAL_SQUARE_SIZE_MM = 20
                px_to_mm_ratio = REAL_SQUARE_SIZE_MM / result
                print(f"Conversion factor: {px_to_mm_ratio:.4f} mm/pixel")
                success = True
                break
        
        if success:
            break 