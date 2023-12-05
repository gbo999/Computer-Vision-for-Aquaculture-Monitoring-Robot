import cv2
import numpy as np

# Assume we have a scale from the previous steps
# scale = ... (real-world units per pixel)

# Let's also assume we have bounding box coordinates for prawns detected by YOLO
# This will be a list of tuples (x_min, y_min, x_max, y_max)
# For example:
# prawn_bboxes = [(50, 100, 200, 250), (300, 400, 450, 500), ...]

def calculate_prawn_sizes(prawn_bboxes, scale):
    prawn_sizes = []
    for bbox in prawn_bboxes:
        x_min, y_min, x_max, y_max = bbox

        # Calculate the width and height of the bounding box in pixels
        bbox_width_px = x_max - x_min
        bbox_height_px = y_max - y_min

        # Convert pixel dimensions to real-world units using the scale
        bbox_width_real = bbox_width_px * scale
        bbox_height_real = bbox_height_px * scale

        # Store the calculated real-world dimensions
        prawn_sizes.append((bbox_width_real, bbox_height_real))
    
    return prawn_sizes

# Example usage
scale = 0.005  # Example scale: 0.005 meters per pixel
prawn_bboxes = [(50, 100, 200, 250), (300, 400, 450, 500)]  # Example bounding boxes

prawn_sizes = calculate_prawn_sizes(prawn_bboxes, scale)

for i, size in enumerate(prawn_sizes):
    width, height = size
    print(f"Prawn {i+1}: Width = {width:.2f} m, Height = {height:.2f} m")
