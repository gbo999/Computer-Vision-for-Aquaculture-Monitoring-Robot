# utils.py
import numpy as np

def parse_pose_estimation(txt_file):
    pose_estimations = []

    with open(txt_file, 'r') as f:
        for line in f:
            pose_estimations.append([float(x) for x in line.strip().split()])
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