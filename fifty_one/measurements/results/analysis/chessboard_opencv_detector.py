import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_chessboard_opencv(image_path, pattern_size=(7, 7), display=True):
    """
    Detect chessboard using OpenCV's built-in findChessboardCorners function
    
    Args:
        image_path: Path to the image file
        pattern_size: Tuple of (rows, cols) internal corners in the chessboard pattern
        display: Whether to display visualization
    
    Returns:
        square_size in pixels if detected, None otherwise
    """
    print(f"Detecting chessboard with OpenCV (pattern: {pattern_size})")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if not found:
        print(f"Could not detect chessboard with pattern size {pattern_size}")
        
        # Try to use the more robust CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(
            gray, 
            pattern_size, 
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if not found:
            # Try with even more robust flags
            found, corners = cv2.findChessboardCorners(
                gray, 
                pattern_size, 
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                      cv2.CALIB_CB_NORMALIZE_IMAGE +
                      cv2.CALIB_CB_FAST_CHECK
            )
            
            if not found:
                if display:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title('Failed to detect chessboard corners')
                    plt.axis('off')
                    plt.show()
                return None
    
    # We found corners, now refine them for better accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Calculate square size by averaging distances between adjacent corners
    square_size = calculate_square_size(corners, pattern_size)
    
    if display:
        # Draw and display the corners
        img_corners = img.copy()
        cv2.drawChessboardCorners(img_corners, pattern_size, corners, found)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Corners\nSquare Size: {square_size:.2f} px')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print(f"Successfully detected chessboard with OpenCV! Square size: {square_size:.2f} pixels")
    return square_size

def calculate_square_size(corners, pattern_size):
    """
    Calculate the average square size from the detected corners
    
    Args:
        corners: Detected chessboard corners from OpenCV
        pattern_size: Tuple of (rows, cols) of the pattern
    
    Returns:
        Average square size in pixels
    """
    # Reshape corners to a 2D array with shape (rows*cols, 2)
    corners = corners.reshape(-1, 2)
    
    # Calculate horizontal distances (along rows)
    h_distances = []
    for row in range(pattern_size[0]):
        for col in range(pattern_size[1] - 1):
            idx1 = row * pattern_size[1] + col
            idx2 = row * pattern_size[1] + col + 1
            x1, y1 = corners[idx1]
            x2, y2 = corners[idx2]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            h_distances.append(distance)
    
    # Calculate vertical distances (along columns)
    v_distances = []
    for col in range(pattern_size[1]):
        for row in range(pattern_size[0] - 1):
            idx1 = row * pattern_size[1] + col
            idx2 = (row + 1) * pattern_size[1] + col
            x1, y1 = corners[idx1]
            x2, y2 = corners[idx2]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            v_distances.append(distance)
    
    # Average of horizontal and vertical distances
    h_avg = np.mean(h_distances)
    v_avg = np.mean(v_distances)
    
    return (h_avg + v_avg) / 2

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect chessboard using OpenCV')
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
    detect_chessboard_opencv(args.image_path, pattern_size) 