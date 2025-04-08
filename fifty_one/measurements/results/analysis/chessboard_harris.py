import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def detect_chessboard_harris(image_path, block_size=3, ksize=5, k=0.04, threshold_ratio=0.15, 
                            min_distance=20, max_corners=200):
    """
    Detect chessboard corners using Harris corner detector with strict filtering.
    
    Args:
        image_path: Path to the chessboard image
        block_size: Size of neighborhood for corner detection
        ksize: Aperture parameter for Sobel derivative
        k: Harris detector free parameter
        threshold_ratio: Ratio of maximum value to use as threshold (higher = stricter)
        min_distance: Minimum distance between corners
        max_corners: Maximum number of corners to detect
        
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
    
    # Apply adaptive thresholding to enhance corners
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Enhance image with Gaussian blur (reduce noise but preserve edges)
    processed = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect corners using Shi-Tomasi (more selective)
    corners = cv2.goodFeaturesToTrack(
        processed, max_corners, qualityLevel=0.1, minDistance=min_distance
    )
    
    if corners is None or len(corners) < 12:  # Need at least a 4x4 grid
        print(f"Not enough strong corners detected. Trying Harris detector with stricter parameters.")
        
        # Use Harris with high threshold
        dst = cv2.cornerHarris(processed, block_size, ksize, k)
        dst_dilated = cv2.dilate(dst, None)
        threshold = threshold_ratio * dst_dilated.max()  # Higher threshold_ratio = fewer corners
        corner_mask = dst_dilated > threshold
        
        # Get corner coordinates and refine them
        harris_corners = np.argwhere(corner_mask)
        if len(harris_corners) < 12:
            print("Not enough corners detected even with Harris detector.")
            return None
            
        # Convert to (x, y) format instead of (row, col)
        harris_corners = np.flip(harris_corners, axis=1).astype(np.float32)
        
        # Use strict non-maximum suppression
        filtered_corners = []
        for i in range(len(harris_corners)):
            x, y = harris_corners[i]
            # Check if this is the maximum in its neighborhood
            is_max = True
            for j in range(len(harris_corners)):
                if i == j:
                    continue
                x2, y2 = harris_corners[j]
                dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                if dist < min_distance and dst_dilated[int(y), int(x)] < dst_dilated[int(y2), int(x2)]:
                    is_max = False
                    break
            if is_max:
                filtered_corners.append(harris_corners[i])
        
        if len(filtered_corners) < 12:
            print(f"Not enough strong corners after filtering ({len(filtered_corners)})")
            return None
            
        # Convert to standard format
        filtered_corners = np.array(filtered_corners).reshape(-1, 1, 2)
        corners = filtered_corners
    
    # Convert corners to a more usable format
    corners = corners.reshape(-1, 2)
    
    # Use DBSCAN to filter outliers and keep only the main grid cluster
    if len(corners) > 16:  # Only cluster if we have enough corners
        clustering = DBSCAN(eps=min_distance*1.5, min_samples=4).fit(corners)
        labels = clustering.labels_
        
        # Keep only corners from the largest cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) > 1 and any(unique_labels >= 0):
            largest_cluster = unique_labels[np.argmax(counts[unique_labels >= 0])]
            corners = corners[labels == largest_cluster]
            print(f"Filtered to {len(corners)} corners in largest cluster")
    
    # Try to identify a grid structure
    # Sort corners by Y then X to attempt to find rows
    y_sorted_idx = np.argsort(corners[:, 1])
    corners_sorted = corners[y_sorted_idx]
    
    # Try to estimate spacing for grid structure
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(8, len(corners)), algorithm='ball_tree').fit(corners)
    distances, indices = nbrs.kneighbors(corners)
    
    # Get median of smallest non-zero distances to estimate grid spacing
    neighbor_distances = distances[:, 1:].flatten()
    neighbor_distances.sort()
    spacing_estimate = np.median(neighbor_distances[:len(corners)*2])
    
    # Now use the spacing to group corners into rows with a tighter tolerance
    tolerance = spacing_estimate * 0.3  # Strict tolerance based on estimated spacing
    rows = []
    current_row = [corners_sorted[0]]
    current_y = corners_sorted[0, 1]
    
    for i in range(1, len(corners_sorted)):
        if abs(corners_sorted[i, 1] - current_y) < tolerance:
            current_row.append(corners_sorted[i])
        else:
            if len(current_row) >= 3:  # Only keep rows with at least 3 corners
                rows.append(np.array(current_row))
            current_row = [corners_sorted[i]]
            current_y = corners_sorted[i, 1]
    
    if len(current_row) >= 3:
        rows.append(np.array(current_row))
    
    # Sort each row by x
    for i in range(len(rows)):
        rows[i] = rows[i][np.argsort(rows[i][:, 0])]
    
    # Filter rows to ensure they all have a similar number of corners (grid consistency)
    row_lengths = [len(row) for row in rows]
    if row_lengths:
        mode_length = max(set(row_lengths), key=row_lengths.count)
        filtered_rows = [row for row in rows if abs(len(row) - mode_length) <= 1]
        
        # Only keep rows that are evenly spaced
        if len(filtered_rows) >= 3:
            # Calculate vertical distances between adjacent rows
            row_centers = [np.mean(row, axis=0)[1] for row in filtered_rows]
            row_distances = [row_centers[i+1] - row_centers[i] for i in range(len(row_centers)-1)]
            
            # Filter to keep only evenly spaced rows
            median_row_dist = np.median(row_distances)
            consistent_rows = [filtered_rows[0]]
            expected_y = row_centers[0]
            
            for i in range(1, len(filtered_rows)):
                expected_y += median_row_dist
                if abs(row_centers[i] - expected_y) < tolerance * 2:
                    consistent_rows.append(filtered_rows[i])
            
            filtered_rows = consistent_rows if len(consistent_rows) >= 3 else filtered_rows
        
        rows = filtered_rows
    
    # Determine if we have a valid grid (at least 3 rows with at least 3 corners each)
    if len(rows) < 3 or any(len(row) < 3 for row in rows):
        print(f"Could not identify a clear grid structure. Found {len(rows)} rows.")
        # Not a valid grid
        return None
    
    # Use the filtered rows to get grid corners
    filtered_corners = np.vstack(rows)
    
    # Calculate the average distance between adjacent corners in the grid
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
    
    # Calculate median of horizontal and vertical distances separately
    # This better handles chessboards where squares aren't perfectly square
    h_median = np.median(h_distances)
    v_median = np.median(v_distances)
    
    # Calculate the square size as the average of horizontal and vertical distances
    avg_square_size = (h_median + v_median) / 2
    
    # Display the grid visually
    img_corners = img.copy()
    # Draw lines to show the detected grid
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
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Grid: {len(rows)}×{min(len(row) for row in rows)}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Grid size: {len(rows)} rows x {min(len(row) for row in rows)} columns")
    print(f"Horizontal spacing: {h_median:.2f} pixels")
    print(f"Vertical spacing: {v_median:.2f} pixels")
    print(f"Average square size: {avg_square_size:.2f} pixels")
    return avg_square_size

def improved_chessboard_detection(image_path, pattern_size=(9,6)):
    """
    Combined approach that first tries OpenCV's findChessboardCorners and falls back to Harris if that fails.
    
    Args:
        image_path: Path to the chessboard image
        pattern_size: Size of the chessboard pattern (corners, not squares)
        
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
    
    # Try standard chessboard corner detection first with strict parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    
    if ret:
        print("OpenCV chessboard detector succeeded!")
        # Refine corner detection with strict criteria
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw the corners
        img_corners = img.copy()
        cv2.drawChessboardCorners(img_corners, pattern_size, corners2, ret)
        
        # Calculate distances between adjacent corners (horizontal and vertical)
        h_distances = []
        v_distances = []
        
        for i in range(len(corners2)):
            row = i // pattern_size[0]
            col = i % pattern_size[0]
            
            # Check right neighbor
            if col < pattern_size[0] - 1:
                right_idx = row * pattern_size[0] + (col + 1)
                dist = np.linalg.norm(corners2[i] - corners2[right_idx])
                h_distances.append(dist)
            
            # Check bottom neighbor
            if row < pattern_size[1] - 1:
                bottom_idx = (row + 1) * pattern_size[0] + col
                dist = np.linalg.norm(corners2[i] - corners2[bottom_idx])
                v_distances.append(dist)
        
        # Calculate median of horizontal and vertical distances separately
        h_median = np.median(h_distances)
        v_median = np.median(v_distances)
        
        # Square size is the average of horizontal and vertical distances
        avg_square_size = (h_median + v_median) / 2
        
        # Display the results
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
        plt.title(f'Chessboard Pattern: {pattern_size[0]}×{pattern_size[1]}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Chessboard pattern: {pattern_size[0]}×{pattern_size[1]} corners")
        print(f"Horizontal spacing: {h_median:.2f} pixels")
        print(f"Vertical spacing: {v_median:.2f} pixels")
        print(f"Average square size: {avg_square_size:.2f} pixels")
        return avg_square_size
    else:
        print("OpenCV chessboard detector failed, trying enhanced Harris corner detector...")
        return detect_chessboard_harris(image_path)

# Example usage:
if __name__ == "__main__":
    # Replace with the actual path to your chessboard image
    image_path = '/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/1/carapace/right/undistorted_GX010068_27_795.jpg_gamma.jpg'
    square_size = improved_chessboard_detection(image_path)
    
    if square_size is not None:
        print(f"Final result: {square_size:.2f} pixels") 