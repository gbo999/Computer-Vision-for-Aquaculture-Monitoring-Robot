import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def detect_chessboard_grid(image_path, visualize=True):
    """
    Detect chessboard grid and calculate square size using the approach from
    https://medium.com/@siromermer/extracting-chess-square-coordinates-dynamically-with-opencv-image-processing-methods-76b933f0f64e
    
    Args:
        image_path: Path to the chessboard image
        visualize: Whether to visualize the results
        
    Returns:
        float: The average size of a square in pixels, or None if detection fails
        dict: Information about the grid including corners and lines
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Check if image was loaded successfully
    if img is None:
        print(f"Error: Could not read image at '{image_path}'. Please check if the path is correct.")
        return None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Step 2: Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Step 3: Apply morphology operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Step 4: Invert the binary image if needed (chessboards usually have dark grid lines)
    # Check percentage of white pixels to determine if we need to invert
    white_percentage = np.sum(binary == 255) / binary.size
    if white_percentage > 0.5:  # If more than 50% is white, invert
        binary = cv2.bitwise_not(binary)
    
    # Step 5: Find all contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 6: Filter contours - keeping only square/rectangular ones
    potential_squares = []
    for contour in contours:
        # Get contour area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Skip very small or very large contours
        if area < 100 or area > img.shape[0] * img.shape[1] * 0.25:
            continue
            
        # Approximate contour to polygon
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Keep only quadrilaterals
        if len(approx) == 4:
            # Check if it's square-like based on aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(float(w) / h, float(h) / w)  # Always >= 1
            
            if aspect_ratio < 1.5:  # Allow slightly rectangular shapes
                potential_squares.append(contour)
    
    if len(potential_squares) < 4:  # Need at least a few squares for a valid chessboard
        print(f"Not enough potential squares detected ({len(potential_squares)})")
        return None, None
    
    # Step 7: Find the outer boundary of the chessboard
    # Method 1: Look for the largest contour from the potential squares
    chessboard_mask = np.zeros_like(binary)
    for square in potential_squares:
        cv2.drawContours(chessboard_mask, [square], -1, 255, -1)
    
    # Apply morphology to connect squares
    kernel = np.ones((5, 5), np.uint8)
    chessboard_mask = cv2.morphologyEx(chessboard_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find the largest connected component
    contours, _ = cv2.findContours(chessboard_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Could not find chessboard boundary")
        return None, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate to get a nice boundary
    perimeter = cv2.arcLength(largest_contour, True)
    chessboard_boundary = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)
    
    # Step 8: Find the Hough lines (horizontal and vertical lines of the grid)
    # Create an edge image for Hough transform
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    
    # Apply mask to focus only on the chessboard region
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [chessboard_boundary], -1, 255, -1)
    edges = cv2.bitwise_and(edges, mask)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None or len(lines) < 10:  # Need enough lines for a chessboard grid
        print(f"Not enough lines detected: {0 if lines is None else len(lines)}")
        return None, None
    
    # Step 9: Separate horizontal and vertical lines
    h_lines = []
    v_lines = []
    
    for line in lines:
        rho, theta = line[0]
        if theta < np.pi * 0.25 or theta > np.pi * 0.75:  # Vertical lines
            v_lines.append((rho, theta))
        else:  # Horizontal lines
            h_lines.append((rho, theta))
    
    # Step 10: Cluster similar lines to avoid duplicates
    def cluster_lines(lines, threshold=30):
        # Convert lines to a format suitable for clustering
        points = np.array([[np.abs(rho)] for rho, theta in lines])
        
        # Cluster lines based on their rho values
        clustering = DBSCAN(eps=threshold, min_samples=1).fit(points)
        labels = clustering.labels_
        
        # Get the representative line for each cluster (median value)
        unique_labels = np.unique(labels)
        clustered_lines = []
        
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            # Get the line with median rho value in this cluster
            median_idx = indices[len(indices) // 2]
            clustered_lines.append(lines[median_idx])
        
        return clustered_lines
    
    # Cluster horizontal and vertical lines separately
    h_lines = cluster_lines(h_lines)
    v_lines = cluster_lines(v_lines)
    
    # Step 11: Make sure we have enough lines in each direction
    if len(h_lines) < 7 or len(v_lines) < 7:  # Need at least 7 lines in each direction for a 6x6 grid
        print(f"Not enough grid lines after clustering: H={len(h_lines)}, V={len(v_lines)}")
        return None, None
    
    # Step 12: Sort lines by position
    h_lines.sort(key=lambda line: line[0])
    v_lines.sort(key=lambda line: line[0])
    
    # Step 13: Calculate line intersections to find grid points
    def line_intersection(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2
        
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([rho1, rho2])
        
        try:
            x, y = np.linalg.solve(A, b)
            return int(x), int(y)
        except np.linalg.LinAlgError:
            return None  # Lines are parallel
    
    # Calculate all grid points
    grid_points = []
    for h_line in h_lines:
        for v_line in v_lines:
            intersection = line_intersection(h_line, v_line)
            if intersection:
                x, y = intersection
                # Check if point is inside the image
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    grid_points.append((x, y))
    
    if len(grid_points) < 16:  # Need at least a 4x4 grid
        print(f"Not enough valid intersections: {len(grid_points)}")
        return None, None
    
    # Step 14: Calculate square size based on adjacent grid points
    # Organize grid points into a 2D array based on their positions
    # First sort by y-coordinate (rows)
    grid_points.sort(key=lambda p: p[1])
    
    # Group points with similar y-coordinates into rows
    from sklearn.cluster import DBSCAN
    
    # Find an appropriate epsilon value based on the data
    points_array = np.array(grid_points)
    y_coords = points_array[:, 1].reshape(-1, 1)
    clustering = DBSCAN(eps=20, min_samples=1).fit(y_coords)
    labels = clustering.labels_
    
    # Group points by row
    rows = []
    for label in np.unique(labels):
        row_points = [grid_points[i] for i in range(len(grid_points)) if labels[i] == label]
        # Sort each row by x-coordinate
        row_points.sort(key=lambda p: p[0])
        rows.append(row_points)
    
    # Sort rows by average y-coordinate
    rows.sort(key=lambda row: sum(p[1] for p in row) / len(row))
    
    # Make sure we have a valid grid structure
    if len(rows) < 2:
        print("Could not organize points into a grid structure")
        return None, None
    
    # Calculate horizontal and vertical distances
    h_distances = []
    v_distances = []
    
    # Horizontal distances within each row
    for row in rows:
        for i in range(len(row) - 1):
            x1, y1 = row[i]
            x2, y2 = row[i + 1]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            h_distances.append(distance)
    
    # Vertical distances between corresponding points in adjacent rows
    for i in range(len(rows) - 1):
        for j in range(min(len(rows[i]), len(rows[i+1]))):
            if j < len(rows[i]) and j < len(rows[i+1]):
                x1, y1 = rows[i][j]
                x2, y2 = rows[i+1][j]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                v_distances.append(distance)
    
    if not h_distances or not v_distances:
        print("Could not calculate grid spacing")
        return None, None
    
    # Calculate median distances to avoid outliers
    h_median = np.median(h_distances)
    v_median = np.median(v_distances)
    
    # Square size is the average of horizontal and vertical spacing
    square_size = (h_median + v_median) / 2
    
    # Step 15: Visualize results if requested
    if visualize:
        img_result = img.copy()
        
        # Draw chessboard boundary
        cv2.drawContours(img_result, [chessboard_boundary], -1, (0, 255, 0), 2)
        
        # Draw grid lines
        for rho, theta in h_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_result, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        for rho, theta in v_lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img_result, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Draw grid points
        for point in grid_points:
            x, y = point
            cv2.circle(img_result, (x, y), 5, (0, 255, 255), -1)
        
        plt.figure(figsize=(16, 8))
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(edges, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Grid: {len(rows)}×{len(rows[0])}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Output results
    print(f"Grid size: approximately {len(rows)}×{len(rows[0])} intersections")
    print(f"Horizontal spacing: {h_median:.2f} pixels")
    print(f"Vertical spacing: {v_median:.2f} pixels")
    print(f"Average square size: {square_size:.2f} pixels")
    
    # Return square size and grid information
    grid_info = {
        'square_size': square_size,
        'h_lines': h_lines,
        'v_lines': v_lines,
        'grid_points': grid_points,
        'rows': rows,
        'boundary': chessboard_boundary
    }
    
    return square_size, grid_info

def main(image_path):
    """
    Main function to detect chessboard grid and calculate square size
    """
    print("=" * 60)
    print("CHESSBOARD GRID DETECTION USING HOUGH LINES APPROACH")
    print("=" * 60)
    
    square_size, grid_info = detect_chessboard_grid(image_path)
    
    if square_size is not None:
        print("\n" + "=" * 50)
        print(f"FINAL RESULT: Chessboard square size: {square_size:.2f} pixels")
        
        # If you know the real-world size of the chessboard squares (e.g., in mm),
        # you can calculate the conversion factor:
        REAL_SQUARE_SIZE_MM = 20  # Replace with actual size in mm
        px_to_mm_ratio = REAL_SQUARE_SIZE_MM / square_size
        print(f"Conversion factor: {px_to_mm_ratio:.4f} mm/pixel")
        print(f"1 pixel = {1/px_to_mm_ratio:.4f} mm")
        print("=" * 50)
        return square_size
    else:
        print("\n" + "=" * 50)
        print("FAILED to detect a valid chessboard grid.")
        print("Suggestions:")
        print("1. Try an image with better lighting and contrast")
        print("2. Make sure the chessboard grid lines are clearly visible")
        print("3. Ensure the chessboard is not severely distorted")
        print("=" * 50)
        return None

if __name__ == "__main__":
    # Replace with the actual path to your chessboard image
    image_path = '/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/1/carapace/right/undistorted_GX010068_27_795.jpg_gamma.jpg'
    main(image_path) 