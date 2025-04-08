import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def detect_chessboard_corners_fast(image_path, threshold=40, nonmax_suppression=True, 
                                   grid_strict=True, min_grid_size=3):
    """
    Lightweight corner detector specifically for chessboard corners using FAST detector.
    
    Args:
        image_path: Path to the chessboard image
        threshold: Threshold for the FAST detector (higher = fewer corners)
        nonmax_suppression: Whether to apply non-maximum suppression
        grid_strict: Whether to strictly enforce grid structure
        min_grid_size: Minimum grid size (rows and columns) to consider valid
        
    Returns:
        float: The average size of a square in pixels, or None if detection fails
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if img is None:
        print(f"Error: Could not read image at '{image_path}'. Please check if the path is correct.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Step 1: Detect potential chessboard squares using adaptive thresholding
    # This helps us focus only on areas with chessboard-like patterns
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to clean up the binary image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Step 2: Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape to identify potential squares
    square_mask = np.zeros_like(binary)
    for contour in contours:
        # Get contour attributes
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Check if the contour is square-like (has 4 corners)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Simple square filter: approximation has 4 points and is somewhat square
        if len(approx) == 4 and area > 100 and perimeter > 40:
            # Check if it's square-like based on aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.7 < aspect_ratio < 1.3:  # Only accept square-like shapes
                cv2.drawContours(square_mask, [contour], -1, 255, -1)
    
    # Step 3: Apply the FAST corner detector only in the regions of interest
    processed = enhanced.copy()
    processed[square_mask == 0] = 0  # Mask out non-square regions
    
    # Initialize the FAST detector
    fast = cv2.FastFeatureDetector_create(threshold=threshold, 
                                         nonmaxSuppression=nonmax_suppression)
    
    # Detect keypoints
    keypoints = fast.detect(processed, None)
    
    if len(keypoints) < 12:  # Need a minimum number of corners
        print(f"Not enough corners detected ({len(keypoints)}). Try lowering the threshold.")
        return None
    
    # Step 4: Convert keypoints to a more usable format
    corners = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    
    # Use DBSCAN to filter outliers and keep only the main grid cluster
    if len(corners) > 16:
        # Calculate appropriate epsilon based on average nearest neighbor distance
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2).fit(corners)
        distances, _ = nbrs.kneighbors(corners)
        avg_distance = np.median(distances[:, 1]) * 2  # Twice the median nearest distance
        
        # Cluster corners
        clustering = DBSCAN(eps=avg_distance, min_samples=4).fit(corners)
        labels = clustering.labels_
        
        # Keep only corners from the largest cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[unique_labels >= 0]
        if len(valid_labels) > 0:
            largest_cluster = valid_labels[np.argmax(counts[unique_labels >= 0])]
            corners = corners[labels == largest_cluster]
            print(f"Filtered to {len(corners)} corners in largest cluster")
    
    # Step 5: Organize corners into a grid structure
    if grid_strict and len(corners) >= 9:  # Need at least a 3x3 grid
        # Sort corners by Y to identify rows
        y_sorted_idx = np.argsort(corners[:, 1])
        corners_sorted = corners[y_sorted_idx]
        
        # Estimate grid spacing
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(8, len(corners)), algorithm='ball_tree').fit(corners)
        distances, _ = nbrs.kneighbors(corners)
        
        # Get median of smallest non-zero distances to estimate grid spacing
        neighbor_distances = distances[:, 1:].flatten()
        neighbor_distances.sort()
        spacing_estimate = np.median(neighbor_distances[:len(corners)*2])
        
        # Group corners into rows using estimated spacing
        tolerance = spacing_estimate * 0.4
        rows = []
        current_row = [corners_sorted[0]]
        current_y = corners_sorted[0, 1]
        
        for i in range(1, len(corners_sorted)):
            if abs(corners_sorted[i, 1] - current_y) < tolerance:
                current_row.append(corners_sorted[i])
            else:
                if len(current_row) >= min_grid_size:
                    rows.append(np.array(current_row))
                current_row = [corners_sorted[i]]
                current_y = corners_sorted[i, 1]
        
        if len(current_row) >= min_grid_size:
            rows.append(np.array(current_row))
        
        # Sort each row by x coordinate
        for i in range(len(rows)):
            rows[i] = rows[i][np.argsort(rows[i][:, 0])]
        
        # Check if we have a valid grid
        if len(rows) < min_grid_size or any(len(row) < min_grid_size for row in rows):
            print(f"Could not identify a clear grid structure. Found {len(rows)} rows.")
            # Still proceed with all corners, but not as a grid
            filtered_corners = corners
            is_grid = False
        else:
            # Use the structured rows
            filtered_corners = np.vstack(rows)
            is_grid = True
    else:
        # No grid enforcement, just use all the corners
        filtered_corners = corners
        is_grid = False
        rows = []
    
    # Step 6: Calculate square size
    if is_grid and len(rows) >= min_grid_size:
        # Calculate horizontal and vertical distances in the grid
        h_distances = []
        v_distances = []
        
        # Horizontal distances (within rows)
        for row in rows:
            if len(row) > 1:
                for i in range(len(row) - 1):
                    h_distances.append(np.linalg.norm(row[i] - row[i+1]))
        
        # Vertical distances (between corresponding points in adjacent rows)
        for i in range(len(rows) - 1):
            min_cols = min(len(rows[i]), len(rows[i+1]))
            for j in range(min_cols):
                v_distances.append(np.linalg.norm(rows[i][j] - rows[i+1][j]))
        
        if not h_distances or not v_distances:
            print("Failed to calculate grid spacing.")
            return None
        
        # Calculate median of horizontal and vertical distances
        h_median = np.median(h_distances)
        v_median = np.median(v_distances)
        
        # Square size is the average of horizontal and vertical distances
        avg_square_size = (h_median + v_median) / 2
    else:
        # No grid - use nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors
        if len(filtered_corners) < 2:
            print("Not enough corners for calculation.")
            return None
            
        nbrs = NearestNeighbors(n_neighbors=min(5, len(filtered_corners)), algorithm='ball_tree').fit(filtered_corners)
        distances, indices = nbrs.kneighbors(filtered_corners)
        
        # Use median of the smallest non-zero distances
        square_size_estimates = distances[:, 1]
        avg_square_size = np.median(square_size_estimates)
    
    # Step 7: Visualize results
    img_corners = img.copy()
    
    # Draw grid lines if we have a grid
    if is_grid:
        for row in rows:
            for i in range(len(row)-1):
                pt1 = tuple(map(int, row[i]))
                pt2 = tuple(map(int, row[i+1]))
                cv2.line(img_corners, pt1, pt2, (0, 255, 0), 1)
        
        # Draw vertical lines
        for j in range(min(len(row) for row in rows)):
            for i in range(len(rows)-1):
                if j < len(rows[i]) and j < len(rows[i+1]):
                    pt1 = tuple(map(int, rows[i][j]))
                    pt2 = tuple(map(int, rows[i+1][j]))
                    cv2.line(img_corners, pt1, pt2, (0, 255, 0), 1)
    
    # Draw corners
    for corner in filtered_corners:
        x, y = corner
        cv2.circle(img_corners, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # Display results
    plt.figure(figsize=(16, 8))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(binary, cmap='gray')
    plt.title('Processed Binary Image')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    if is_grid:
        plt.title(f'Detected Grid: {len(rows)}×{min(len(row) for row in rows)}')
    else:
        plt.title(f'Detected {len(filtered_corners)} Corners')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Output results
    if is_grid:
        print(f"Grid size: {len(rows)} rows x {min(len(row) for row in rows)} columns")
        print(f"Horizontal spacing: {h_median:.2f} pixels")
        print(f"Vertical spacing: {v_median:.2f} pixels")
    else:
        print(f"Detected {len(filtered_corners)} corners (no clear grid)")
    
    print(f"Average square size: {avg_square_size:.2f} pixels")
    return avg_square_size

def try_multiple_thresholds(image_path, thresholds=[20, 30, 40, 50, 60], 
                           nonmax_suppression=True, grid_strict=True):
    """
    Try multiple threshold values for the FAST corner detector.
    
    Args:
        image_path: Path to the chessboard image
        thresholds: List of threshold values to try
        nonmax_suppression: Whether to apply non-maximum suppression
        grid_strict: Whether to strictly enforce grid structure
        
    Returns:
        float: The average size of a square in pixels from the best threshold
    """
    best_result = None
    best_threshold = None
    
    for threshold in thresholds:
        print(f"\nTrying FAST detector with threshold: {threshold}")
        result = detect_chessboard_corners_fast(
            image_path, 
            threshold=threshold,
            nonmax_suppression=nonmax_suppression,
            grid_strict=grid_strict
        )
        
        if result is not None and result > 5:
            best_result = result
            best_threshold = threshold
            print(f"Success with threshold {threshold}!")
            break
    
    if best_result is not None:
        print(f"\nBest threshold: {best_threshold}")
        print(f"Final square size: {best_result:.2f} pixels")
    else:
        print("\nFailed to detect chessboard corners with any threshold.")
    
    return best_result

# Example usage
if __name__ == "__main__":
    # Replace with the actual path to your chessboard image
    image_path = '/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/1/carapace/right/undistorted_GX010068_27_795.jpg_gamma.jpg'
    
    # Try with multiple thresholds
    square_size = try_multiple_thresholds(image_path)
    
    if square_size is not None:
        # If you know the real-world size of the squares
        REAL_SQUARE_SIZE_MM = 20  # Replace with actual size in mm
        px_to_mm_ratio = REAL_SQUARE_SIZE_MM / square_size
        print(f"Conversion factor: {px_to_mm_ratio:.4f} mm/pixel") 