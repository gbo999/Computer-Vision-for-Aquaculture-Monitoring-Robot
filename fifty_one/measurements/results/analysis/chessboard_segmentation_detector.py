import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_chessboard_with_segmentation(image_path, pattern_size=(7, 7), display=True):
    """
    Detect chessboard using color segmentation approach
    
    Args:
        image_path: Path to the image file
        pattern_size: Tuple of (rows, cols) of the chessboard pattern
        display: Whether to display visualization
    
    Returns:
        square_size in pixels if detected, None otherwise
    """
    print(f"Detecting chessboard with segmentation method (pattern: {pattern_size})")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to HSV to better segment by color/brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Try multiple HSV ranges to segment the chessboard
    hsv_ranges = [
        # Lighter squares
        (np.array([0, 0, 150]), np.array([179, 50, 255])),
        # Darker squares
        (np.array([0, 0, 30]), np.array([179, 110, 150])),
        # Medium squares
        (np.array([0, 0, 80]), np.array([179, 70, 200]))
    ]
    
    for lower_hsv, upper_hsv in hsv_ranges:
        # Create binary mask
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Try to find chessboard corners in the binary mask
        ret, corners = cv2.findChessboardCorners(mask, pattern_size, 
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                               cv2.CALIB_CB_NORMALIZE_IMAGE +
                                               cv2.CALIB_CB_FILTER_QUADS)
        
        if ret:
            # Refine corner detection
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(mask, corners, (11, 11), (-1, -1), criteria)
            
            # Calculate square size
            corners = corners.reshape(-1, 2)
            h_distances = []
            v_distances = []
            
            for i in range(len(corners)):
                row = i // pattern_size[0]
                col = i % pattern_size[0]
                
                if col < pattern_size[0] - 1:
                    right_idx = row * pattern_size[0] + (col + 1)
                    dist = np.linalg.norm(corners[i] - corners[right_idx])
                    h_distances.append(dist)
                
                if row < pattern_size[1] - 1:
                    bottom_idx = (row + 1) * pattern_size[0] + col
                    dist = np.linalg.norm(corners[i] - corners[bottom_idx])
                    v_distances.append(dist)
            
            square_size = (np.median(h_distances) + np.median(v_distances)) / 2
            
            if display:
                # Draw detected corners
                img_corners = img.copy()
                cv2.drawChessboardCorners(img_corners, pattern_size, corners.reshape(-1, 1, 2), ret)
                
                # Display results
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title('Original Image')
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap='gray')
                plt.title('Binary Mask')
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
                plt.title(f'Detected Corners\nSquare Size: {square_size:.2f} px')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
            
            print(f"Successfully detected chessboard! Square size: {square_size:.2f} pixels")
            return square_size
    
    print("Could not detect chessboard with segmentation method")
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect chessboard using segmentation method')
    parser.add_argument('image_path', help='Path to the chessboard image')
    parser.add_argument('--pattern', default='7,7', help='Pattern size as rows,cols (e.g. 7,7)')
    
    args = parser.parse_args()
    
    # Parse pattern size
    parts = args.pattern.split(',')
    if len(parts) == 2:
        pattern_size = (int(parts[0]), int(parts[1]))
    else:
        pattern_size = (7, 7)
        print(f"Invalid pattern format: {args.pattern}. Using default (7,7).")
    
    # Run detection
    detect_chessboard_with_segmentation(args.image_path, pattern_size) 