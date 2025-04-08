import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path

# Import all our detection methods
from chessboard_segmentation_detector import detect_chessboard_with_segmentation
from chessboard_grid_detector import detect_chessboard_grid
from fast_corner_detector import detect_chessboard_corners_fast
from chessboard_harris import detect_chessboard_harris

def standard_opencv_detection(image_path, pattern_size=(7, 7)):
    """Standard OpenCV chessboard detection"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, 
                                           cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                           cv2.CALIB_CB_NORMALIZE_IMAGE +
                                           cv2.CALIB_CB_FILTER_QUADS)
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Calculate square size
        square_size = calculate_square_size(corners, pattern_size)
        
        # Draw corners for visualization
        img_corners = img.copy()
        cv2.drawChessboardCorners(img_corners, pattern_size, corners, ret)
        
        return square_size, img_corners
    
    return None, None

def calculate_square_size(corners, pattern_size):
    """Calculate square size from corners"""
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
    
    return (np.median(h_distances) + np.median(v_distances)) / 2

def test_all_methods(image_path, pattern_sizes=[(7, 7), (9, 6), (8, 5), (6, 6)]):
    """Test all chessboard detection methods"""
    print(f"\nAnalyzing image: {os.path.basename(image_path)}")
    print("=" * 70)
    
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Create figure for results
    plt.figure(figsize=(15, 10))
    
    # Display original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # Dictionary to store results
    results = {}
    
    # 1. Standard OpenCV detection
    print("\n1. TESTING STANDARD OPENCV DETECTION")
    best_square_size = None
    best_pattern = None
    best_corners_img = None
    
    for pattern in pattern_sizes:
        print(f"  Trying pattern {pattern}...")
        start_time = time.time()
        square_size, corners_img = standard_opencv_detection(image_path, pattern)
        elapsed = time.time() - start_time
        
        if square_size is not None:
            print(f"  SUCCESS! Square size: {square_size:.2f} pixels (Pattern: {pattern})")
            print(f"  Time: {elapsed:.3f} seconds")
            best_square_size = square_size
            best_pattern = pattern
            best_corners_img = corners_img
            break
            
    if best_square_size is None:
        print("  FAILED: Could not detect with standard OpenCV method")
    else:
        results['OpenCV Standard'] = {
            'square_size': best_square_size,
            'pattern': best_pattern,
            'time': elapsed
        }
        
        # Display results
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(best_corners_img, cv2.COLOR_BGR2RGB))
        plt.title(f'OpenCV Standard\nPattern: {best_pattern}\n{best_square_size:.2f} px')
        plt.axis('off')
    
    # 2. StackOverflow Segmentation method
    print("\n2. TESTING SEGMENTATION-BASED DETECTION (StackOverflow)")
    best_square_size = None
    best_pattern = None
    
    for pattern in pattern_sizes:
        print(f"  Trying pattern {pattern}...")
        start_time = time.time()
        square_size = detect_chessboard_with_segmentation(image_path, pattern, display=False)
        elapsed = time.time() - start_time
        
        if square_size is not None:
            print(f"  SUCCESS! Square size: {square_size:.2f} pixels (Pattern: {pattern})")
            print(f"  Time: {elapsed:.3f} seconds")
            best_square_size = square_size
            best_pattern = pattern
            break
    
    if best_square_size is None:
        print("  FAILED: Could not detect with segmentation method")
    else:
        results['Segmentation'] = {
            'square_size': best_square_size,
            'pattern': best_pattern,
            'time': elapsed
        }
        
        # Rerun to get visualization
        img = cv2.imread(image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([179, 70, 200]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        plt.subplot(2, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Segmentation Method\nPattern: {best_pattern}\n{best_square_size:.2f} px')
        plt.axis('off')

    # 3. Harris Corner method
    print("\n3. TESTING HARRIS CORNER DETECTION")
    start_time = time.time()
    corners, square_size, threshold = detect_chessboard_harris(image_path, display=False)
    elapsed = time.time() - start_time
    
    if square_size is not None:
        print(f"  SUCCESS! Square size: {square_size:.2f} pixels")
        print(f"  Time: {elapsed:.3f} seconds")
        results['Harris'] = {
            'square_size': square_size,
            'threshold': threshold,
            'time': elapsed
        }
        
        # Create visualization
        img = cv2.imread(image_path)
        for corner in corners:
            cv2.circle(img, (int(corner[0]), int(corner[1])), 5, (0, 255, 0), -1)
            
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Harris Corner Method\n{square_size:.2f} px')
        plt.axis('off')
    else:
        print("  FAILED: Could not detect with Harris method")
    
    # 4. FAST Corner method
    print("\n4. TESTING FAST CORNER DETECTION")
    start_time = time.time()
    corners, square_size = detect_chessboard_corners_fast(image_path, display=False)
    elapsed = time.time() - start_time
    
    if square_size is not None:
        print(f"  SUCCESS! Square size: {square_size:.2f} pixels")
        print(f"  Time: {elapsed:.3f} seconds")
        results['FAST'] = {
            'square_size': square_size,
            'time': elapsed
        }
        
        # Create visualization
        img = cv2.imread(image_path)
        for corner in corners:
            cv2.circle(img, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)
            
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'FAST Corner Method\n{square_size:.2f} px')
        plt.axis('off')
    else:
        print("  FAILED: Could not detect with FAST method")
    
    # 5. Grid Detection (Hough Lines) method
    print("\n5. TESTING GRID DETECTION (Hough Lines)")
    start_time = time.time()
    square_size = detect_chessboard_grid(image_path, display=False)
    elapsed = time.time() - start_time
    
    if square_size is not None:
        print(f"  SUCCESS! Square size: {square_size:.2f} pixels")
        print(f"  Time: {elapsed:.3f} seconds")
        results['Grid'] = {
            'square_size': square_size,
            'time': elapsed
        }
        
        # Run again to get visualization
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        plt.subplot(2, 3, 6)
        plt.imshow(edges, cmap='gray')
        plt.title(f'Grid Detection\n{square_size:.2f} px')
        plt.axis('off')
    else:
        print("  FAILED: Could not detect with Grid method")
    
    # Show summary results
    plt.tight_layout()
    plt.savefig(f"chessboard_detection_comparison_{Path(image_path).stem}.png")
    plt.show()
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)
    
    if results:
        # Calculate average square size from successful methods
        square_sizes = [data['square_size'] for data in results.values()]
        avg_square_size = np.mean(square_sizes)
        std_square_size = np.std(square_sizes)
        
        print(f"Average square size: {avg_square_size:.2f} ± {std_square_size:.2f} pixels")
        print("\nResults by method:")
        
        for method, data in results.items():
            print(f"  {method}: {data['square_size']:.2f} px (in {data['time']:.3f} sec)")
            
        # Calculate conversion factor
        REAL_SQUARE_SIZE_MM = 20  # Replace with the actual size
        px_to_mm_ratio = REAL_SQUARE_SIZE_MM / avg_square_size
        
        print("\nMeasurement conversion factors:")
        print(f"  {px_to_mm_ratio:.6f} mm/pixel")
        print(f"  {1/px_to_mm_ratio:.6f} pixels/mm")
    else:
        print("No successful detection with any method.")
        
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare different chessboard detection methods')
    parser.add_argument('image_path', help='Path to the chessboard image')
    parser.add_argument('--patterns', nargs='+', type=str, default=['7,7', '9,6', '8,5', '6,6'],
                        help='Pattern sizes to try, format: rows,cols (e.g. 7,7)')
    
    args = parser.parse_args()
    
    # Parse pattern sizes
    pattern_sizes = []
    for pattern_str in args.patterns:
        parts = pattern_str.split(',')
        if len(parts) == 2:
            try:
                pattern = (int(parts[0]), int(parts[1]))
                pattern_sizes.append(pattern)
            except ValueError:
                print(f"Invalid pattern format: {pattern_str}. Should be rows,cols")
    
    if not pattern_sizes:
        pattern_sizes = [(7, 7), (9, 6), (8, 5), (6, 6)]
    
    # Run test
    test_all_methods(args.image_path, pattern_sizes) 