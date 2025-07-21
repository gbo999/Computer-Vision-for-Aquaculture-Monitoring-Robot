import numpy as np
import cv2

# Camera calibration parameters
camera_matrix = np.array([
    [3015.014085286132, 0.0, 1596.4745000545388],
    [0.0, 3015.014085286132, 1596.4745000545388],
    [0.0, 0.0, 1.0]
])

dist_coeffs = np.array([
    0.2528090891288297,
    0.1324221379663344,
    0.07048721428221141,
    0.1610412047411928
])

def undistort_labels(labels_text):
    """
    Undistort YOLO format labels using camera calibration parameters.
    Each line in labels_text should be in YOLO format: class x_center y_center width height kp1_x kp1_y kp1_conf kp2_x kp2_y kp2_conf ...
    """
    undistorted_labels = []
    
    for line in labels_text.strip().split('\n'):
        if not line.strip():
            continue
            
        values = [float(x) for x in line.split()]
        if len(values) != 17:  # 1 class + 4 bbox + 4 keypoints × 3 values
            continue
            
        # Extract class and bounding box
        class_id = values[0]
        x_center, y_center, width, height = values[1:5]
        
        # Convert YOLO format to absolute coordinates
        x1 = x_center - width/2
        y1 = y_center - height/2
        x2 = x_center + width/2
        y2 = y_center + height/2
        
        # Convert normalized coordinates to pixel coordinates
        img_width = 3192  # Based on camera matrix
        img_height = 3192  # Based on camera matrix
        
        x1_px = x1 * img_width
        y1_px = y1 * img_height
        x2_px = x2 * img_width
        y2_px = y2 * img_height
        
        # Create points array for undistortion
        points = np.array([
            [x1_px, y1_px],
            [x2_px, y2_px]
        ]).reshape(-1, 1, 2)
        
        # Undistort bounding box corners
        undistorted_points = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
        
        # Convert back to normalized coordinates
        x1_undist, y1_undist = undistorted_points[0][0]
        x2_undist, y2_undist = undistorted_points[1][0]
        
        x1_undist_norm = x1_undist / img_width
        y1_undist_norm = y1_undist / img_height
        x2_undist_norm = x2_undist / img_width
        y2_undist_norm = y2_undist / img_height
        
        # Calculate new center and dimensions in normalized coordinates
        new_x_center = (x1_undist_norm + x2_undist_norm) / 2
        new_y_center = (y1_undist_norm + y2_undist_norm) / 2
        new_width = x2_undist_norm - x1_undist_norm
        new_height = y2_undist_norm - y1_undist_norm
        
        # Process keypoints
        keypoints = []
        for i in range(5, len(values), 3):
            kp_x, kp_y, kp_conf = values[i:i+3]
            if kp_conf != 2:  # Skip if confidence is not 2
                keypoints.extend([kp_x, kp_y, kp_conf])
                continue
                
            # Convert keypoint to pixel coordinates
            kp_x_px = kp_x * img_width
            kp_y_px = kp_y * img_height
            
            # Undistort keypoint
            kp_point = np.array([[kp_x_px, kp_y_px]]).reshape(-1, 1, 2)
            undistorted_kp = cv2.undistortPoints(kp_point, camera_matrix, dist_coeffs, P=camera_matrix)
            
            # Convert back to normalized coordinates
            kp_x_undist = undistorted_kp[0][0][0] / img_width
            kp_y_undist = undistorted_kp[0][0][1] / img_height
            
            keypoints.extend([kp_x_undist, kp_y_undist, kp_conf])
        
        # Create new label line
        new_label = [class_id, new_x_center, new_y_center, new_width, new_height] + keypoints
        undistorted_labels.append(' '.join(map(str, new_label)))
    
    return '\n'.join(undistorted_labels)

def process_file(input_path, output_path):
    """
    Process a file containing YOLO format labels and save the undistorted results.
    """
    with open(input_path, 'r') as f:
        labels = f.read()
    
    undistorted_labels = undistort_labels(labels)
    
    with open(output_path, 'w') as f:
        f.write(undistorted_labels)

if __name__ == "__main__":
    # Example usage
    input_file = "src/measurement/notebooks/output.txt"
    output_file = "src/measurement/notebooks/undistorted_output.txt"
    process_file(input_file, output_file) 