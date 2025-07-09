# utils.py
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import os

def parse_pose_estimation(txt_file: str) -> List[List[float]]:
    """
    Parse YOLO format pose estimation data from a text file.
    
    The function handles the YOLOv8 keypoint label format:
        class xc yc w h kp1x kp1y kp1v kp2x kp2y kp2v ...
    
    Where:
    - class: class ID (0 for prawn)
    - xc, yc: center coordinates of bounding box (normalized [0-1])
    - w, h: width and height of bounding box (normalized [0-1])
    - kpNx, kpNy: keypoint coordinates (normalized [0-1])
    - kpNv: keypoint visibility/confidence score [0-1]
    
    Keypoint order:
    0: start_carapace
    1: eyes
    2: rostrum
    3: tail
    
    Args:
        txt_file (str): Path to the YOLO format annotation file
        
    Returns:
        List[List[float]]: List of pose estimations, where each pose is a list of floats:
            [class_id, x_center, y_center, width, height, kp1_x, kp1_y, kp1_conf, ...]
            
    Raises:
        FileNotFoundError: If txt_file does not exist
        ValueError: If a line contains invalid data
        
    Example:
        >>> poses = parse_pose_estimation("path/to/labels.txt")
        >>> # First pose, first keypoint x-coordinate (normalized)
        >>> start_carapace_x = poses[0][5]
    """
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Annotation file not found: {txt_file}")
        
    pose_estimations = []
    unique_lines = set()  # Track unique lines to prevent duplicates
    
    try:
        with open(txt_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines or duplicates
                if not line or line in unique_lines:
                    continue
                    
                unique_lines.add(line)
                
                try:
                    # Parse all values as floats
                    values = [float(x) for x in line.split()]
                    
                    # Validate number of values
                    # Minimum: class + bbox (5 values)
                    # Each keypoint adds 3 values (x, y, conf)
                    if len(values) < 5:
                        print(f"Warning: Line {line_num} has insufficient values: {line}")
                        continue
                        
                    # Validate keypoint triplets
                    num_keypoints = (len(values) - 5) // 3
                    if (len(values) - 5) % 3 != 0:
                        print(f"Warning: Line {line_num} has incomplete keypoint triplet: {line}")
                        continue
                        
                    # Validate value ranges
                    for i, val in enumerate(values):
                        if i == 0:  # class_id should be integer
                            if not val.is_integer():
                                print(f"Warning: Line {line_num} has non-integer class ID: {val}")
                        else:  # all other values should be in [0, 1]
                            if not 0 <= val <= 1:
                                print(f"Warning: Line {line_num} has out-of-range value: {val}")
                    
                    pose_estimations.append(values)
                    
                except ValueError as e:
                    print(f"Warning: Failed to parse line {line_num}: {line}")
                    print(f"Error: {str(e)}")
                    continue
                    
    except Exception as e:
        print(f"Error reading pose estimation file {txt_file}: {str(e)}")
        return []
        
    return pose_estimations

def calculate_euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_real_width(focal_length, distance_to_object, width_in_pixels, pixel_size):
    width_in_sensor = width_in_pixels * pixel_size
    real_width_mm = (width_in_sensor * distance_to_object) / focal_length
    return real_width_mm

def extract_identifier_from_gt(filename):
    return filename.split('-')[0]

def calculate_bbox_area(bbox):
    """
    Calculate the area of a bounding box.
    Bounding box is expected to be a tuple in the format (x_min, y_min, x_max, y_max).
    """
    x_min, y_min, x_max, y_max = bbox
    return (x_max - x_min) * (y_max - y_min) 