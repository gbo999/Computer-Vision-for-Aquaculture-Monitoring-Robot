import os
from pathlib import Path
import math
import shutil
import cv2
import numpy as np
from tqdm import tqdm






def calculate_euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_size_error(total_length_mm, expected_total):
    """
    Calculate how far the total length is from the expected value.
    Returns a normalized error score (lower is better).
    """
    # Calculate percentage error
    return abs(total_length_mm - expected_total) / expected_total

def filter_detection_by_size(total_length_mm, expected_total, is_big=False):
    """
    Filter detections based on total length.
    Big: fixed range 160-220mm
    Small: 20% tolerance from expected
    Returns whether it matches and the error score (lower is better).
    """
    if is_big:
        # Fixed range for big exuviae
        total_min = 175  # 16.0cm
        total_max = 220  # 22.0cm
    else:
        # Percentage tolerance for small exuviae
        tolerance_percent = 20
        total_min = expected_total * (1 - tolerance_percent/100)
        total_max = expected_total * (1 + tolerance_percent/100)
    
    # Check if measurement falls within range
    total_ok = total_min <= total_length_mm <= total_max
    
    # Calculate error score
    error_score = calculate_size_error(total_length_mm, expected_total)
    
    return total_ok, error_score

def determine_size(total_length_mm):
    """
    Determine if a detection is big or small based on total length only.
    Big: 160-220mm fixed range
    Small: 145mm ± 20%
    """
    # Expected values
    big_total = 180    # 18.0cm (but using fixed range 16-22cm)
    small_total = 145  # 14.5cm (using ±20% tolerance)
    
    # Check both size possibilities
    is_big, big_error = filter_detection_by_size(
        total_length_mm, expected_total=big_total, is_big=True
    )
    
    is_small, small_error = filter_detection_by_size(
        total_length_mm, expected_total=small_total, is_big=False
    )
    
    if is_big and is_small:
        # If it matches both, choose big (since it's in the big range)
        return ("big", big_error)
    elif is_big:
        return ("big", big_error)
    elif is_small:
        return ("small", small_error)
    else:
        return ("rejected", float('inf'))

def filter_predictions(input_dir, output_dir):
    """
    Filter YOLO format predictions based on size criteria.
    
    Args:
        input_dir: Directory containing the original label files
        output_dir: Directory where filtered label files will be saved
    """
    # Constants for size calculation (original image dimensions)
    calc_width = 5312
    calc_height = 2988
    
    # Constants for display (resized image dimensions)
    display_width = 640
    display_height = 360
    
    # Camera parameters
    diagonal_fov = 84.6
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each label file
    label_files = list(Path(input_dir).glob("*.txt"))
    print(f"Processing {len(label_files)} label files...")
    
    filtered_counts = {"big": 0, "small": 0, "rejected": 0}
    all_lengths = []  # Store all measurements for analysis
    
    for label_file in tqdm(label_files, desc="Processing label files"):
        # Store detections by category with their errors
        big_candidates = []
        small_candidates = []
        
        # Determine image type and height
        filename = label_file.name
        is_circle2 = "GX010191" in filename
        height_mm = 700 if is_circle2 else 410
        
        # Read detections
        with open(label_file, 'r') as f:
            detections = f.readlines()
        
        print(f"\nProcessing {filename} (height: {height_mm}mm)")
        
        for detection in detections:
            values = list(map(float, detection.strip().split()))
            class_id = int(values[0])
            
            # Extract keypoints (x, y coordinates)
            keypoints = []
            for i in range(5, len(values)-1, 3):  # Skip class, bbox coords, and confidences
                x = values[i]  # These are normalized coordinates (0-1)
                y = values[i + 1]
                keypoints.append([x, y])
            
            if len(keypoints) >= 4:  # Ensure we have all required keypoints
                # Convert normalized coordinates to original image pixels for measurement
                keypoints_calc = []
                for kp in keypoints:
                    x = kp[0] * calc_width
                    y = kp[1] * calc_height
                    keypoints_calc.append([x, y])
                
                # Calculate total length in original image pixels
                total_length_pixels = calculate_euclidean_distance(keypoints_calc[3], keypoints_calc[2])     # tail to rostrum
                
                # Convert to mm using original image dimensions
                diagonal_image_size = math.sqrt(calc_width ** 2 + calc_height ** 2)
                total_length_mm = (total_length_pixels / diagonal_image_size) * (2 * height_mm * math.tan(math.radians(diagonal_fov/2)))
                all_lengths.append(total_length_mm)
                
                # Determine size category based only on total length
                size_category, error = determine_size(total_length_mm)
                
                print(f"  Length: {total_length_mm:.1f}mm -> {size_category} (error: {error:.3f})")
                
                # Store candidates by category
                if size_category == "big":
                    big_candidates.append((detection, error))
                elif size_category == "small":
                    small_candidates.append((detection, error))
                else:
                    filtered_counts["rejected"] += 1
        
        # Select best candidates (one per category if available)
        filtered_detections = []
        
        # Add best big candidate if available
        if big_candidates:
            best_big = min(big_candidates, key=lambda x: x[1])
            filtered_detections.append(best_big[0])
            filtered_counts["big"] += 1
            print(f"  Selected big with error: {best_big[1]:.3f}")
        
        # Add best small candidate if available
        if small_candidates:
            best_small = min(small_candidates, key=lambda x: x[1])
            filtered_detections.append(best_small[0])
            filtered_counts["small"] += 1
            print(f"  Selected small with error: {best_small[1]:.3f}")
        
        # Save filtered detections
        output_file = Path(output_dir) / label_file.name
        with open(output_file, 'w') as f:
            f.writelines(filtered_detections)
    
    # Print summary statistics
    all_lengths = np.array(all_lengths)
    print("\nMeasurement Statistics:")
    print(f"Min length: {np.min(all_lengths):.1f}mm")
    print(f"Max length: {np.max(all_lengths):.1f}mm")
    print(f"Mean length: {np.mean(all_lengths):.1f}mm")
    print(f"Median length: {np.median(all_lengths):.1f}mm")
    print(f"Std dev: {np.std(all_lengths):.1f}mm")
    
    # Print filtering ranges
    print("\nFiltering Ranges:")
    print("Big: 160-220mm")
    print("Small: 116-174mm")
    
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