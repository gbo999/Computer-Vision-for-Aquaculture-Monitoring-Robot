import pandas as pd
import math
from pathlib import Path
import numpy as np

def calculate_euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def filter_detection_by_size(total_length_mm, carapace_length_mm, expected_total, expected_carapace, tolerance_percent=20):
    """
    Filter detections based on expected sizes with a tolerance range.
    
    Args:
        total_length_mm: Measured total length in mm
        carapace_length_mm: Measured carapace length in mm
        expected_total: Expected total length in mm
        expected_carapace: Expected carapace length in mm
        tolerance_percent: Allowed percentage deviation from expected values
        
    Returns:
        bool: True if detection passes the filter, False otherwise
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

def analyze_measurements(csv_path):
    """
    Analyze measurements from detections and compare with expected values.
    
    Expected values:
    - big_tot = 18cm
    - big_carapace = 6.3cm
    - small_tot = 14.5cm
    - small_carapace = 4.1cm
    
    Keypoint indices:
    0: start_carapace
    1: eyes
    2: rostrum
    3: tail
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Constants
    image_width = 5312
    image_height = 2988
    horizontal_fov = 75.2
    vertical_fov = 46
    
    results = []
    
    for idx, row in df.iterrows():
        # Convert string representation of poses to list
        poses = eval(str(row['object_poses']))
        
        # Extract keypoints (assuming YOLO format: x, y, conf for each point)
        keypoints = []
        for i in range(0, len(poses)-1, 3):
            x = poses[i] * image_width
            y = poses[i + 1] * image_height
            keypoints.append([x, y])
        
        # Set parameters based on image type and size class
        if 'circle2' in csv_path and row['which'] == 'big':
            height_mm = 700  # For right images
            expected_total = 180  # 18cm in mm
            expected_carapace = 63  # 6.3cm in mm
        elif 'square' in csv_path and row['which'] == 'big':
            height_mm = 410  # For square/left images
            expected_total = 180  # 18cm in mm
            expected_carapace = 63  # 6.3cm in mm
        elif 'square' in csv_path and row['which'] == 'small':
            height_mm = 410  # For square/left images
            expected_total = 145  # 14.5cm in mm
            expected_carapace = 41  # 4.1cm in mm
        elif 'circle2' in csv_path and row['which'] == 'small':
            height_mm = 700  # For right images
            expected_total = 145  # 14.5cm in mm
            expected_carapace = 41  # 4.1cm in mm
        
        # Calculate real-world distances
        total_length_pixels = calculate_euclidean_distance(keypoints[3], keypoints[2])     # tail to rostrum
        carapace_length_pixels = calculate_euclidean_distance(keypoints[0], keypoints[1])  # start_carapace to eyes
        
        # Convert to mm using camera parameters
        diagonal_fov = 84.6
        diagonal_image_size = math.sqrt(image_width ** 2 + image_height ** 2)
        
        total_length_mm = (total_length_pixels / diagonal_image_size) * (2 * height_mm * math.tan(math.radians(diagonal_fov/2)))
        carapace_length_mm = (carapace_length_pixels / diagonal_image_size) * (2 * height_mm * math.tan(math.radians(diagonal_fov/2)))
        
        # Filter detection based on size
        passes_filter = filter_detection_by_size(
            total_length_mm, 
            carapace_length_mm,
            expected_total,
            expected_carapace
        )
        
        if passes_filter:
            results.append({
                'image_name': row['image_name'],
                'which': row['which'],
                'pond': csv_path.split('/')[-1].split('_')[-1],
                'total_length_mm': total_length_mm,
                'carapace_length_mm': carapace_length_mm,
                'expected_total_mm': expected_total,
                'expected_carapace_mm': expected_carapace,
                'total_diff_mm': (total_length_mm - expected_total),
                'carapace_diff_mm': (carapace_length_mm - expected_carapace)
            })
    
    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = Path(csv_path).parent / f'filtered_measurement_analysis_{csv_path.split("/")[-1].split("_")[-1]}'
    results_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print("\nFiltered Measurement Analysis Summary:")
    print(f"Total detections that passed size filter: {len(results_df)}")
    if len(results_df) > 0:
        print("\nAverage differences (mm):")
        print(f"Total length: {results_df['total_diff_mm'].mean():.2f}")
        print(f"Carapace length: {results_df['carapace_diff_mm'].mean():.2f}")
        print("\nStandard deviations (mm):")
        print(f"Total length: {results_df['total_diff_mm'].std():.2f}")
        print(f"Carapace length: {results_df['carapace_diff_mm'].std():.2f}")
    
    return results_df

# Analyze both CSV files and save them to diffe
ight_results = analyze_measurements("fifty_one/measurements/filter_detections_stats_circle2.csv")
square_results = analyze_measurements("fifty_one/measurements/filter_detections_stats_square.csv") 