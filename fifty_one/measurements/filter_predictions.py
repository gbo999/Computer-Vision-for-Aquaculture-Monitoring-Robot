import os
from pathlib import Path
import math
import shutil

def calculate_euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def filter_detection_by_size(total_length_mm, carapace_length_mm, expected_total, expected_carapace, tolerance_percent=20):
    """
    Filter detections based on expected sizes with a tolerance range.
    """
    # Calculate allowed ranges
    total_min = expected_total * (1 - tolerance_percent/100)
    total_max = expected_total * (1 + tolerance_percent/100)
    carapace_min = expected_carapace * (1 - tolerance_percent/100)
    carapace_max = expected_carapace * (1 + tolerance_percent/100)
    
    # Check if measurements fall within ranges
    total_ok = total_min <= total_length_mm <= total_max
    carapace_ok = carapace_min <= carapace_length_mm <= carapace_max
    
    return total_ok and carapace_ok

def filter_predictions(input_dir, output_dir):
    """
    Filter YOLO format predictions based on size criteria.
    
    Args:
        input_dir: Directory containing the original label files
        output_dir: Directory where filtered label files will be saved
    """
    # Constants
    image_width = 5312
    image_height = 2988
    diagonal_fov = 84.6
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each label file
    label_files = list(Path(input_dir).glob("*.txt"))
    print(f"Processing {len(label_files)} label files...")
    
    filtered_counts = {"big": 0, "small": 0, "rejected": 0}
    
    for label_file in label_files:
        filtered_detections = []
        
        # Determine image type and height
        filename = label_file.name
        is_circle2 = "GX010194" in filename  # Adjust these patterns based on your naming convention
        height_mm = 700 if is_circle2 else 410
        
        # Read detections
        with open(label_file, 'r') as f:
            detections = f.readlines()
        
        for detection in detections:
            values = list(map(float, detection.strip().split()))
            class_id = int(values[0])
            
            # Extract keypoints (x, y coordinates)
            keypoints = []
            for i in range(5, len(values)-1, 3):  # Skip class, bbox coords, and confidences
                x = values[i] * image_width
                y = values[i + 1] * image_height
                keypoints.append([x, y])
            
            if len(keypoints) >= 4:  # Ensure we have all required keypoints
                # Calculate distances
                total_length_pixels = calculate_euclidean_distance(keypoints[3], keypoints[2])     # tail to rostrum
                carapace_length_pixels = calculate_euclidean_distance(keypoints[0], keypoints[1])  # start_carapace to eyes
                
                # Convert to mm
                diagonal_image_size = math.sqrt(image_width ** 2 + image_height ** 2)
                total_length_mm = (total_length_pixels / diagonal_image_size) * (2 * height_mm * math.tan(math.radians(diagonal_fov/2)))
                carapace_length_mm = (carapace_length_pixels / diagonal_image_size) * (2 * height_mm * math.tan(math.radians(diagonal_fov/2)))
                
                # Check both size possibilities
                is_big = filter_detection_by_size(total_length_mm, carapace_length_mm, 180, 63)  # 18cm total, 6.3cm carapace
                is_small = filter_detection_by_size(total_length_mm, carapace_length_mm, 145, 41)  # 14.5cm total, 4.1cm carapace
                
                if is_big:
                    filtered_detections.append(detection)
                    filtered_counts["big"] += 1
                elif is_small:
                    filtered_detections.append(detection)
                    filtered_counts["small"] += 1
                else:
                    filtered_counts["rejected"] += 1
        
        # Save filtered detections
        output_file = Path(output_dir) / label_file.name
        with open(output_file, 'w') as f:
            f.writelines(filtered_detections)
    
    # Print summary
    print("\nFiltering Summary:")
    print(f"Large exuviae detected: {filtered_counts['big']}")
    print(f"Small exuviae detected: {filtered_counts['small']}")
    print(f"Rejected detections: {filtered_counts['rejected']}")
    print(f"\nFiltered label files saved to: {output_dir}")

if __name__ == "__main__":
    input_dir = "runs/pose/predict57/labels"
    output_dir = "runs/pose/predict57/filtered_labels"
    filter_predictions(input_dir, output_dir) 