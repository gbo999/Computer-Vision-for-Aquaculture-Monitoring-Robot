import os
import cv2
import glob
import numpy as np
from collections import defaultdict

# Directories
img_dir = r'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/runs/pose/predict51'
labels_dir = os.path.join(img_dir, 'labels')
output_dir = 'detection_examples'

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Dictionary to store detections by video group
groups = defaultdict(list)

# First pass: collect all detections and their sizes
for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
    basename = os.path.basename(label_file)
    video_group = basename.split('_')[1]  # Get GX010191, etc.
    img_file = os.path.join(img_dir, basename.replace('.txt', '.jpg'))
    
    if not os.path.exists(img_file):
        continue
        
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:  # Need class_id + bbox + 4 keypoints with conf
                continue
            
            # Store detection info
            detection = {
                'img_file': img_file,
                'bbox': [float(x) for x in parts[1:5]],  # x_center, y_center, width, height
                'keypoints': [],
                'area': float(parts[3]) * float(parts[4])  # width * height
            }
            
            # Store keypoints
            for i in range(4):
                kp_x = float(parts[5 + i*3])
                kp_y = float(parts[6 + i*3])
                conf = float(parts[7 + i*3])
                detection['keypoints'].append((kp_x, kp_y, conf))
                
            groups[video_group].append(detection)

def draw_detection(img, detection, color=(0, 255, 0)):
    h, w = img.shape[:2]
    x_center, y_center, width, height = detection['bbox']
    
    # Convert relative coordinates to absolute
    x1 = int((x_center - width/2) * w)
    y1 = int((y_center - height/2) * h)
    x2 = int((x_center + width/2) * w)
    y2 = int((y_center + height/2) * h)
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw size information
    text = f"w: {width:.3f}, h: {height:.3f}"
    cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw keypoints
    for kp_x, kp_y, conf in detection['keypoints']:
        if conf > 0.5:
            kp_x = int(kp_x * w)
            kp_y = int(kp_y * h)
            cv2.circle(img, (kp_x, kp_y), 3, (255, 0, 0), -1)

# For each video group, show examples at different size ranges
for video_group, detections in groups.items():
    print(f"\nProcessing {video_group}...")
    
    # Sort detections by area
    detections.sort(key=lambda x: x['area'])
    
    # Select examples from different size ranges
    n_examples = 5  # Number of examples per group
    step = len(detections) // n_examples
    
    for i in range(0, len(detections), step):
        if i >= len(detections):
            break
            
        detection = detections[i]
        img = cv2.imread(detection['img_file'])
        if img is None:
            continue
            
        # Get coordinates for cropping
        h, w = img.shape[:2]
        x_center, y_center, width, height = detection['bbox']
        
        # Convert to pixel coordinates and add padding
        padding = 100
        x1 = max(0, int((x_center - width/2) * w) - padding)
        y1 = max(0, int((y_center - height/2) * h) - padding)
        x2 = min(w, int((x_center + width/2) * w) + padding)
        y2 = min(h, int((y_center + height/2) * h) + padding)
        
        # Crop and draw
        crop = img[y1:y2, x1:x2].copy()
        
        # Adjust detection coordinates for cropped image
        detection_crop = detection.copy()
        detection_crop['bbox'][0] = (detection['bbox'][0] * w - x1) / (x2 - x1)
        detection_crop['bbox'][1] = (detection['bbox'][1] * h - y1) / (y2 - y1)
        detection_crop['bbox'][2] = detection['bbox'][2] * w / (x2 - x1)
        detection_crop['bbox'][3] = detection['bbox'][3] * h / (y2 - y1)
        
        for j in range(len(detection_crop['keypoints'])):
            kp = detection_crop['keypoints'][j]
            detection_crop['keypoints'][j] = (
                (kp[0] * w - x1) / (x2 - x1),
                (kp[1] * h - y1) / (y2 - y1),
                kp[2]
            )
        
        draw_detection(crop, detection_crop)
        
        # Save the example
        output_file = os.path.join(output_dir, f"{video_group}_example_{i//step+1}.jpg")
        cv2.imwrite(output_file, crop)
        print(f"Saved example {i//step+1} for {video_group}")
        print(f"Width: {width:.4f}, Height: {height:.4f}, Area: {width*height:.4f}")

print(f"\nExample detections have been saved to {output_dir}/")
print("Each image shows the detection with its dimensions (width and height in relative coordinates)")
print("Green boxes: Detections")
print("Blue dots: Keypoints") 