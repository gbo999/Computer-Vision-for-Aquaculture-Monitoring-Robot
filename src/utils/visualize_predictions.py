import os
import cv2
import glob
import numpy as np

# Directories
img_dir = 'runs/predict/keypoint_detection'
orig_labels_dir = 'runs/predict/keypoint_detection/labels'
filtered_labels_dir = 'runs/predict/keypoint_detection/filtered_labels'
output_dir = 'runs/predict/keypoint_detection/visualization'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Colors for visualization (BGR format)
ORIGINAL_COLOR = (0, 0, 255)    # Red for original predictions
FILTERED_COLOR = (0, 255, 0)    # Green for filtered predictions
KEYPOINT_COLOR = (255, 0, 0)    # Blue for keypoints

def draw_predictions(img, predictions, color):
    h, w = img.shape[:2]
    for pred in predictions:
        parts = pred.strip().split()
        if len(parts) < 17:  # Skip invalid lines (need class_id + bbox + 4 keypoints with conf)
            continue
            
        # Extract bounding box
        x_center, y_center = float(parts[1]) * w, float(parts[2]) * h
        width, height = float(parts[3]) * w, float(parts[4]) * h
        x1 = int(x_center - width/2)
        y1 = int(y_center - height/2)
        x2 = int(x_center + width/2)
        y2 = int(y_center + height/2)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw keypoints
        for i in range(4):
            kp_x = int(float(parts[5 + i*3]) * w)
            kp_y = int(float(parts[6 + i*3]) * h)
            conf = float(parts[7 + i*3])
            if conf > 0.5:  # Only draw high confidence keypoints
                cv2.circle(img, (kp_x, kp_y), 3, KEYPOINT_COLOR, -1)

def process_image(img_path):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not load image: {img_path}")
        return
        
    # Get corresponding label files
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    orig_label_path = os.path.join(orig_labels_dir, base_name + '.txt')
    filtered_label_path = os.path.join(filtered_labels_dir, base_name + '.txt')
    
    # Create two copies of the image
    img_orig = img.copy()
    img_comparison = img.copy()
    
    # Draw original predictions
    if os.path.exists(orig_label_path):
        with open(orig_label_path, 'r') as f:
            orig_preds = f.readlines()
        draw_predictions(img_orig, orig_preds, ORIGINAL_COLOR)
        draw_predictions(img_comparison, orig_preds, ORIGINAL_COLOR)
    
    # Draw filtered predictions
    if os.path.exists(filtered_label_path):
        with open(filtered_label_path, 'r') as f:
            filtered_preds = f.readlines()
        draw_predictions(img_comparison, filtered_preds, FILTERED_COLOR)
    
    # Save visualizations
    output_orig = os.path.join(output_dir, f"{base_name}_original.jpg")
    output_comparison = os.path.join(output_dir, f"{base_name}_comparison.jpg")
    
    cv2.imwrite(output_orig, img_orig)
    cv2.imwrite(output_comparison, img_comparison)

# Process all images
for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
    process_image(img_path)

print(f"Visualization complete. Results saved in: {output_dir}")
print("Red boxes: Original predictions")
print("Green boxes: Filtered predictions")
print("Blue dots: Keypoints") 