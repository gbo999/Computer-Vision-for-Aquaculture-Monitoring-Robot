import cv2
import numpy as np
import os

def visualize_keypoints_with_numbers(image_path, annotation_path):
    """
    Draw keypoints with their index numbers on the image.
    
    Args:
        image_path: Path to the image file
        annotation_path: Path to the YOLO format annotation file
    """
    # Read image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Read annotations
    with open(annotation_path, 'r') as f:
        line = f.readline().strip()
        values = [float(x) for x in line.split()]
    
    # Extract keypoints
    keypoints = []
    for i in range(5, len(values), 3):  # Start from index 5 (after bbox), step by 3
        x, y, conf = values[i:i+3]
        # Convert normalized coordinates to pixel coordinates
        x_px = int(x * width)
        y_px = int(y * height)
        keypoints.append((x_px, y_px))
    
    # Draw keypoints with numbers
    for idx, (x, y) in enumerate(keypoints):
        # Draw a circle for each keypoint
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red circle
        
        # Draw the keypoint number
        cv2.putText(img, str(idx), (x+10, y+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green number
    
    # Save the visualization
    output_path = image_path.replace('.jpg', '_numbered_keypoints.jpg')
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to: {output_path}")
    
    return output_path

# Example usage
image_path = "/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/test-car/images/GX010152_36_378-jpg_gamma_jpg.rf.d49b41f3c5a08c7aa8fd8a1779b49804.jpg"
annotation_path = "/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/runs/pose/predict3/labels/GX010152_36_378-jpg_gamma_jpg.rf.d49b41f3c5a08c7aa8fd8a1779b49804.txt"

visualize_keypoints_with_numbers(image_path, annotation_path) 