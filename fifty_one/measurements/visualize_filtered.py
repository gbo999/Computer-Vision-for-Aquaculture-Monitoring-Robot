import cv2
import numpy as np
from pathlib import Path
import math

def calculate_euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def determine_size(total_length_mm):
    """
    Determine if a detection is big or small based on total length only.
    Big: 175-220mm          
    Small: 116-174mm
    """
    # Big: fixed range
    if 175 <= total_length_mm <= 220:
        return "BIG", (0, 255, 0)  # Green for big
    
    # Small: percentage range
    small_expected = 145
    small_min = small_expected * 0.8  # -20%
    small_max = small_expected * 1.2  # +20%
    if small_min <= total_length_mm <= small_max:
        return "SMALL", (255, 0, 0)  # Red for small
    
    return "REJECTED", (128, 128, 128)  # Gray for rejected

def draw_detection(img, detection, display_width, display_height, calc_width, calc_height, height_mm):
    """Draw a single detection with keypoints and size label"""
    values = list(map(float, detection.strip().split()))
    
    # Extract bounding box
    x_center, y_center, width, height = values[1:5]
    x1 = int((x_center - width/2) * display_width)
    y1 = int((y_center - height/2) * display_height)
    x2 = int((x_center + width/2) * display_width)
    y2 = int((y_center + height/2) * display_height)
    
    # Extract keypoints
    keypoints = []
    for i in range(5, len(values)-1, 3):
        x = values[i]
        y = values[i + 1]
        conf = values[i + 2]
        keypoints.append([x, y])
    
    if len(keypoints) >= 4:
        # Calculate total length for size determination
        keypoints_calc = []
        for kp in keypoints:
            x = kp[0] * calc_width
            y = kp[1] * calc_height
            keypoints_calc.append([x, y])
        
        # Calculate total length in original image pixels
        total_length_pixels = calculate_euclidean_distance(keypoints_calc[3], keypoints_calc[2])  # tail to rostrum
        
        # Convert to mm using original image dimensions
        diagonal_image_size = math.sqrt(calc_width ** 2 + calc_height ** 2)
        total_length_mm = (total_length_pixels / diagonal_image_size) * (2 * height_mm * math.tan(math.radians(84.6/2)))
        
        # Determine size and color
        size_label, color = determine_size(total_length_mm)
        
        # Add length to label
        size_label = f"{size_label} ({total_length_mm:.0f}mm)"
    else:
        size_label, color = "REJECTED", (128, 128, 128)
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw keypoints and connections
    keypoint_pairs = [(0, 1), (1, 2), (2, 3)]  # Connections between keypoints
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x * display_width), int(y * display_height)
        cv2.circle(img, (x, y), 3, color, -1)  # Reduced circle size for smaller images
        
        # Draw keypoint number (smaller font for smaller images)
        cv2.putText(img, str(i), (x+3, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    # Draw connections between keypoints
    for start_idx, end_idx in keypoint_pairs:
        start_point = (int(keypoints[start_idx][0] * display_width), 
                      int(keypoints[start_idx][1] * display_height))
        end_point = (int(keypoints[end_idx][0] * display_width), 
                    int(keypoints[end_idx][1] * display_height))
        cv2.line(img, start_point, end_point, color, 1)  # Thinner lines for smaller images
    
    # Draw size label (adjusted position and size for smaller images)
    label_x = x1
    label_y = y1 - 5 if y1 > 10 else y1 + 10
    cv2.putText(img, size_label, (label_x, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def visualize_filtered_detections(labels_dir, images_dir, output_dir):
    """
    Draw filtered detections on images with size labels.
    
    Args:
        labels_dir: Directory containing the filtered label files
        images_dir: Directory containing the original images
        output_dir: Directory where visualized images will be saved
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Constants for size calculation (original image dimensions)
    calc_width = 5312
    calc_height = 2988
    
    # Constants for display (resized image dimensions)
    display_width = 640
    display_height = 360
    
    # Process each label file
    label_files = list(Path(labels_dir).glob("*.txt"))
    print(f"Processing {len(label_files)} images...")
    
    # Base directories for both segmented and original images
    segmented_dir = Path("/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized/segmented")
    original_dir = Path("/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized")
    
    for label_file in label_files:
        # Get base image name (remove segmented_ prefix if present)
        base_name = label_file.stem
        if base_name.startswith("segmented_"):
            base_name = base_name[len("segmented_"):]
        image_name = base_name + ".jpg"
        
        # Try both segmented and original paths
        segmented_path = segmented_dir / f"segmented_{image_name}"
        original_path = original_dir / image_name
        
        # Process both images if they exist
        for img_path, prefix in [(segmented_path, "viz_segmented_"), (original_path, "viz_")]:
            if not img_path.exists():
                print(f"Warning: Image {img_path} not found")
                continue
            
            # Read image and detections
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            with open(label_file, 'r') as f:
                detections = f.readlines()
            
            # Determine image type and height
            is_circle2 = "GX010191" in image_name
            height_mm = 700 if is_circle2 else 410
            
            # Draw each detection
            for detection in detections:
                draw_detection(img, detection, display_width, display_height, calc_width, calc_height, height_mm)
            
            # Add legend (adjusted for smaller images)
            legend_y = 15
            cv2.putText(img, "BIG", (5, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(img, "SMALL", (5, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(img, "REJECTED", (5, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            # Save the visualized image with appropriate prefix
            output_file = output_path / f"{prefix}{image_name}"
            cv2.imwrite(str(output_file), img)
            print(f"Saved: {output_file}")
    
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    labels_dir = "runs/pose/predict57/filtered_labels"
    images_dir = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized/segmented"
    output_dir = "runs/pose/predict57/visualized"
    visualize_filtered_detections(labels_dir, images_dir, output_dir) 