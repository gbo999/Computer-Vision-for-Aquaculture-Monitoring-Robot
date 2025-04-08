import os
import numpy as np

# Import our detector implementations
from fifty_one.measurements.results.analysis.chessboard_grid_detector import detect_chessboard_grid
from fifty_one.measurements.results.analysis.chessboard_opencv_detector import detect_chessboard_opencv

def detect_chessboard(image_path, pattern_size=(7, 7), display=False, method="auto"):
    """
    Detect chessboard in an image using specified method
    
    Args:
        image_path: Path to the image file
        pattern_size: Tuple of (rows, cols) internal corners in the chessboard pattern
        display: Whether to display visualization
        method: Detection method - "opencv", "grid", or "auto" (tries both)
    
    Returns:
        dict with: 
            - square_size: Square size in pixels
            - method: Method that succeeded
            - success: Whether detection was successful
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist")
        return {"success": False, "square_size": None, "method": None}
    
    result = {"success": False, "square_size": None, "method": None}
    
    if method.lower() == "opencv" or method.lower() == "auto":
        # Try OpenCV method first
        print(f"Attempting chessboard detection with OpenCV...")
        square_size = detect_chessboard_opencv(image_path, pattern_size, display)
        
        if square_size is not None:
            result = {"success": True, "square_size": square_size, "method": "opencv"}
            if method.lower() == "opencv":
                return result
    
    if method.lower() == "grid" or (method.lower() == "auto" and not result["success"]):
        # Try grid-based method
        print(f"Attempting chessboard detection with Hough grid method...")
        square_size = detect_chessboard_grid(image_path, pattern_size, display)
        
        if square_size is not None:
            result = {"success": True, "square_size": square_size, "method": "grid"}
    
    # Return final result - will contain the successful detection if any
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect chessboard using multiple methods')
    parser.add_argument('image_path', help='Path to the chessboard image')
    parser.add_argument('--pattern', default='7,7', help='Pattern size as rows,cols (e.g. 7,7)')
    parser.add_argument('--display', action='store_true', help='Display visualization')
    parser.add_argument('--method', default='auto', choices=['auto', 'opencv', 'grid'], 
                        help='Detection method to use')
    
    args = parser.parse_args()
    
    # Parse pattern size
    parts = args.pattern.split(',')
    if len(parts) == 2:
        pattern_size = (int(parts[0]), int(parts[1]))
    else:
        pattern_size = (7, 7)
        print(f"Invalid pattern format: {args.pattern}. Using default (7,7).")
    
    # Run detection
    result = detect_chessboard(args.image_path, pattern_size, args.display, args.method)
    
    if result["success"]:
        print(f"Chessboard detection successful!")
        print(f"Method: {result['method']}")
        print(f"Square size: {result['square_size']:.2f} pixels")
    else:
        print(f"Failed to detect chessboard with any method.") 