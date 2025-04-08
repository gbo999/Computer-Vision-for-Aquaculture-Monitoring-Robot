import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_chessboard_grid(image_path, pattern_size=(7, 7), display=True):
    """
    Detect chessboard using Hough line detection to find the grid pattern
    
    Args:
        image_path: Path to the image file
        pattern_size: Tuple of (rows, cols) of the chessboard pattern
        display: Whether to display visualization
    
    Returns:
        square_size in pixels if detected, None otherwise
    """
    print(f"Detecting chessboard with grid/Hough method (pattern: {pattern_size})")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Dilate edges to make lines more visible
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
    
    if lines is None or len(lines) < 10:  # Need sufficient lines for a chessboard
        print("Not enough lines detected for a chessboard")
        return None
    
    # Classify lines as horizontal or vertical based on their angle
    h_lines = []
    v_lines = []
    
    for line in lines:
        rho, theta = line[0]
        # Horizontal lines: near 0 or 180 degrees
        if (theta < 0.3 or theta > np.pi - 0.3):
            h_lines.append((rho, theta))
        # Vertical lines: near 90 degrees
        elif (np.pi/2 - 0.3 < theta < np.pi/2 + 0.3):
            v_lines.append((rho, theta))
    
    # Not enough horizontal or vertical lines
    if len(h_lines) < pattern_size[1] + 1 or len(v_lines) < pattern_size[0] + 1:
        print(f"Not enough grid lines detected: h_lines={len(h_lines)}, v_lines={len(v_lines)}")
        
        if display:
            # Visualize what was detected
            img_lines = img.copy()
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(edges, cmap='gray')
            plt.title('Edge Detection')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
            plt.title('Detected Lines')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return None
    
    # Sort lines by rho value to find evenly spaced grid lines
    h_lines.sort(key=lambda x: x[0])
    v_lines.sort(key=lambda x: x[0])
    
    # Calculate distances between adjacent lines
    h_distances = [abs(h_lines[i+1][0] - h_lines[i][0]) for i in range(len(h_lines)-1)]
    v_distances = [abs(v_lines[i+1][0] - v_lines[i][0]) for i in range(len(v_lines)-1)]
    
    # Use median distance to estimate square size
    h_square_size = np.median(h_distances)
    v_square_size = np.median(v_distances)
    square_size = (h_square_size + v_square_size) / 2
    
    # Find line intersections to locate grid corners
    corners = []
    for h_rho, h_theta in h_lines:
        for v_rho, v_theta in v_lines:
            # Calculate intersection
            A = np.array([
                [np.cos(h_theta), np.sin(h_theta)],
                [np.cos(v_theta), np.sin(v_theta)]
            ])
            b = np.array([h_rho, v_rho])
            try:
                x, y = np.linalg.solve(A, b)
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    corners.append((x, y))
            except np.linalg.LinAlgError:
                # Lines are parallel or nearly parallel
                continue
    
    # If we need to match OpenCV's format of pattern_size
    if len(corners) >= pattern_size[0] * pattern_size[1]:
        # More sophisticated approach would be to sort/filter corners to match expected pattern
        # For now, we'll just verify we have enough corners
        
        if display:
            # Visualize detected grid
            img_grid = img.copy()
            
            # Draw horizontal lines
            for rho, theta in h_lines[:pattern_size[1]+1]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_grid, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw vertical lines
            for rho, theta in v_lines[:pattern_size[0]+1]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(img_grid, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw corners
            for x, y in corners[:pattern_size[0] * pattern_size[1]]:
                cv2.circle(img_grid, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(edges, cmap='gray')
            plt.title('Edge Detection')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
            plt.title(f'Detected Grid\nSquare Size: {square_size:.2f} px')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        print(f"Successfully detected chessboard grid! Square size: {square_size:.2f} pixels")
        return square_size
    
    print("Could not detect complete chessboard grid pattern")
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect chessboard using grid method')
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
    detect_chessboard_grid(args.image_path, pattern_size) 