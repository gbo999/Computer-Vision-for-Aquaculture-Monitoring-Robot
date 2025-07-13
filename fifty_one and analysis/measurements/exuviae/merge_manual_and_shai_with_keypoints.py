import pandas as pd
import numpy as np
import fiftyone as fo
import ast
import math

class ObjectLengthMeasurer:
    """Calculate real-world measurements from pixel coordinates"""
    def __init__(self, image_width, image_height, horizontal_fov, vertical_fov, distance_mm):
        self.image_width = image_width
        self.image_height = image_height
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.distance_mm = distance_mm
        self.scale_x, self.scale_y = self.calculate_scaling_factors()

    def calculate_scaling_factors(self):
        fov_x_rad = math.radians(self.horizontal_fov)
        fov_y_rad = math.radians(self.vertical_fov)
        scale_x = (2 * self.distance_mm * math.tan(fov_x_rad / 2)) / self.image_width
        scale_y = (2 * self.distance_mm * math.tan(fov_y_rad / 2)) / self.image_height
        return scale_x, scale_y

    def normalize_angle(self, angle):
        theta_norm = min(abs(angle % 180), 180 - abs(angle % 180))
        return theta_norm

    def compute_length(self, predicted_length, angle_deg):
        angle_rad = math.radians(angle_deg)
        combined_scale = math.sqrt((self.scale_x * math.cos(angle_rad)) ** 2 + 
                                 (self.scale_y * math.sin(angle_rad)) ** 2)
        length_mm = predicted_length * combined_scale
        return length_mm, combined_scale

    def compute_length_two_points(self, point1, point2):
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        distance_px = math.sqrt(delta_x ** 2 + delta_y ** 2)
        
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
        normalized_angle = self.normalize_angle(angle_deg)
        
        distance_mm, combined_scale = self.compute_length(distance_px, normalized_angle)
        return distance_mm, normalized_angle, distance_px, combined_scale

def calculate_euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_prawn_measurements(points, manual_size, image_name):
    """Calculate total length and carapace length based on keypoints and classification"""
    if not points or len(points) < 4:
        return None, None
    
    # Check if we have valid keypoints (not NaN)
    valid_points = []
    for i, point in enumerate(points):
        if len(point) >= 2 and not np.isnan(point[0]) and not np.isnan(point[1]):
            valid_points.append((i, point))
    
    if len(valid_points) < 4:
        return None, None
    
    # Convert normalized coordinates to pixel coordinates
    calc_width = 5312
    calc_height = 2988
    
    keypoints_calc = []
    for _, point in valid_points:
        x = point[0] * calc_width
        y = point[1] * calc_height
        keypoints_calc.append([x, y])
    
    # Determine pond type and height
    is_circle = "GX010191" in image_name
    
    # Set height based on pond type and manual classification
    if is_circle:  # Circle pond
        if manual_size == 'big':
            height_mm = 660
        elif manual_size == 'small':
            height_mm = 680
        else:
            height_mm = 700  # default
    else:  # Square pond
        if manual_size == 'big':
            height_mm = 370
        elif manual_size == 'small':
            height_mm = 390
        else:
            height_mm = 410  # default
    
    # Create measurer
    measurer = ObjectLengthMeasurer(5312, 2988, 75, 46, height_mm)
    
    # Calculate total length (rostrum to tail - keypoints[2] to keypoints[3])
    total_length_mm, _, _, _ = measurer.compute_length_two_points(
        keypoints_calc[2], keypoints_calc[3])
    
    # Calculate carapace length (start_carapace to eyes - keypoints[0] to keypoints[1])
    carapace_length_mm, _, _, _ = measurer.compute_length_two_points(
        keypoints_calc[0], keypoints_calc[1])
    
    return total_length_mm, carapace_length_mm

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in [x, y, width, height] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def find_matching_keypoints(sample, bbox_normalized):
    """Find keypoints that match the given bounding box by calculating which keypoints fall within the bbox"""
    if not hasattr(sample, 'keypoints') or sample.keypoints is None:
        return None
    
    if not sample.keypoints.keypoints:
        return None
    
    best_match = None
    best_score = 0.0
    
    # Check each keypoint to see which one best matches the bounding box
    for keypoint in sample.keypoints.keypoints:
        if not keypoint.points or len(keypoint.points) < 4:
            continue
            
        points = keypoint.points
        # Count how many keypoints fall within the bounding box
        points_inside = 0
        valid_points = 0
        
        for point in points:
            if len(point) >= 2 and not np.isnan(point[0]) and not np.isnan(point[1]):
                valid_points += 1
                x, y = point[0], point[1]
                
                # Check if point is inside the bounding box
                if (bbox_normalized[0] <= x <= bbox_normalized[0] + bbox_normalized[2] and
                    bbox_normalized[1] <= y <= bbox_normalized[1] + bbox_normalized[3]):
                    points_inside += 1
        
        # Calculate score as percentage of valid points inside the bbox
        if valid_points > 0:
            score = points_inside / valid_points
            if score > best_score:
                best_score = score
                best_match = {
                    'points': points,
                    'score': score,
                    'points_inside': points_inside,
                    'valid_points': valid_points
                }
    
    return best_match

# Load datasets
print("Loading datasets...")
manual_df = pd.read_csv("spreadsheet_files/manual_classifications_updated_20250713_135258.csv")
shai_df = pd.read_csv("spreadsheet_files/Results-shai-exuviae.csv")
dataset = fo.load_dataset("prawn_keypoints")

print(f"Manual classifications: {len(manual_df)}")
print(f"Shai measurements: {len(shai_df)}")
print(f"FiftyOne samples: {len(dataset)}")

# Clean up Shai's image names to match manual classification format
shai_df['image_name'] = shai_df['Label'].str.replace('Shai - exuviae:', '')
# Remove 'colored_' prefix if present and add .jpg extension
shai_df['image_name'] = shai_df['image_name'].str.replace('colored_', '')
shai_df['image_name'] = shai_df['image_name'] + '.jpg'

# Convert Shai's bounding boxes to normalized coordinates (pixel to normalized)
img_width_px = 5312
img_height_px = 2988

shai_df['bbox_x_norm'] = shai_df['BX'] / img_width_px
shai_df['bbox_y_norm'] = shai_df['BY'] / img_height_px
shai_df['bbox_width_norm'] = shai_df['Width'] / img_width_px
shai_df['bbox_height_norm'] = shai_df['Height'] / img_height_px

# Filter manual classifications to only classified ones
manual_classified = manual_df[manual_df['manual_size'] != 'untagged'].copy()

print(f"Manual classified detections: {len(manual_classified)}")

# Merge datasets
merged_data = []

# Process each manual classification
for _, manual_row in manual_classified.iterrows():
    image_name = manual_row['image_name']
    manual_bbox = [manual_row['bbox_x'], manual_row['bbox_y'], 
                   manual_row['bbox_width'], manual_row['bbox_height']]
    
    # Find matching Shai measurement
    shai_matches = shai_df[shai_df['image_name'] == image_name]
    
    best_shai_match = None
    best_iou = 0.0
    
    if len(shai_matches) > 0:
        print(f"Found {len(shai_matches)} Shai measurements for {image_name}")
        
    for _, shai_row in shai_matches.iterrows():
        shai_bbox = [shai_row['bbox_x_norm'], shai_row['bbox_y_norm'],
                     shai_row['bbox_width_norm'], shai_row['bbox_height_norm']]
        
        iou = calculate_iou(manual_bbox, shai_bbox)
        print(f"  Manual bbox: {manual_bbox}")
        print(f"  Shai bbox: {shai_bbox}")
        print(f"  IoU: {iou}")
        
        if iou > best_iou:
            best_iou = iou
            best_shai_match = shai_row
    
    # Find corresponding FiftyOne sample
    base_name = image_name.replace('colored_', '')
    fo_sample = None
    for sample in dataset:
        if base_name in sample.filepath:
            fo_sample = sample
            break
    
    # Get keypoints information and calculate measurements
    keypoints_info = None
    total_length_mm = None
    carapace_length_mm = None
    
    if fo_sample is not None:
        keypoints_match = find_matching_keypoints(fo_sample, manual_bbox)
        if keypoints_match:
            points = keypoints_match['points']
            
            # Calculate measurements based on manual classification and pond type
            total_length_mm, carapace_length_mm = calculate_prawn_measurements(
                points, manual_row['manual_size'], image_name)
            
            keypoints_info = {
                'start_carapace_x': points[0][0] if len(points) > 0 and not np.isnan(points[0][0]) else None,
                'start_carapace_y': points[0][1] if len(points) > 0 and not np.isnan(points[0][1]) else None,
                'eyes_x': points[1][0] if len(points) > 1 and not np.isnan(points[1][0]) else None,
                'eyes_y': points[1][1] if len(points) > 1 and not np.isnan(points[1][1]) else None,
                'rostrum_x': points[2][0] if len(points) > 2 and not np.isnan(points[2][0]) else None,
                'rostrum_y': points[2][1] if len(points) > 2 and not np.isnan(points[2][1]) else None,
                'tail_x': points[3][0] if len(points) > 3 and not np.isnan(points[3][0]) else None,
                'tail_y': points[3][1] if len(points) > 3 and not np.isnan(points[3][1]) else None,
                'keypoints_score': keypoints_match['score'],
                'keypoints_inside_bbox': keypoints_match['points_inside'],
                'keypoints_valid_total': keypoints_match['valid_points'],
                'calculated_total_length_mm': total_length_mm,
                'calculated_carapace_length_mm': carapace_length_mm
            }
    
    # Create merged entry
    merged_entry = {
        # Manual classification data
        'image_name': image_name,
        'manual_size': manual_row['manual_size'],
        'manual_bbox_x': manual_row['bbox_x'],
        'manual_bbox_y': manual_row['bbox_y'],
        'manual_bbox_width': manual_row['bbox_width'],
        'manual_bbox_height': manual_row['bbox_height'],
        'manual_bbox_x_mm': manual_row['bbox_x_mm'],
        'manual_bbox_y_mm': manual_row['bbox_y_mm'],
        'manual_bbox_width_mm': manual_row['bbox_width_mm'],
        'manual_bbox_height_mm': manual_row['bbox_height_mm'],
        
        # Shai measurement data (if matched)
        'shai_matched': best_shai_match is not None,
        'shai_iou': best_iou if best_shai_match is not None else 0.0,
        'shai_length': best_shai_match['Length'] if best_shai_match is not None else None,
        'shai_bbox_x': best_shai_match['BX'] if best_shai_match is not None else None,
        'shai_bbox_y': best_shai_match['BY'] if best_shai_match is not None else None,
        'shai_bbox_width': best_shai_match['Width'] if best_shai_match is not None else None,
        'shai_bbox_height': best_shai_match['Height'] if best_shai_match is not None else None,
        
        # Keypoints data (if available)
        'keypoints_available': keypoints_info is not None,
        **({} if keypoints_info is None else keypoints_info)
    }
    
    merged_data.append(merged_entry)

# Create merged DataFrame
merged_df = pd.DataFrame(merged_data)

# Save merged data
output_path = "spreadsheet_files/merged_manual_shai_keypoints.csv"
merged_df.to_csv(output_path, index=False)

print(f"\nMerged data saved to {output_path}")
print(f"Total merged entries: {len(merged_df)}")

# Print statistics
print("\nMerging Statistics:")
print(f"Manual classifications with Shai matches: {len(merged_df[merged_df['shai_matched']])}")
print(f"Manual classifications with keypoints: {len(merged_df[merged_df['keypoints_available']])}")
print(f"Average IoU for Shai matches: {merged_df[merged_df['shai_matched']]['shai_iou'].mean():.3f}")

print("\nSize distribution:")
print(merged_df['manual_size'].value_counts())

print("\nShai match rate by size:")
print(merged_df.groupby('manual_size')['shai_matched'].mean())

print("\nKeypoints availability by size:")
print(merged_df.groupby('manual_size')['keypoints_available'].mean()) 