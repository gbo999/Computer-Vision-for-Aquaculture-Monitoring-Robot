# data_loader.py
import fiftyone as fo
import pandas as pd
import os
import ast
from tqdm import tqdm
from fifty_one.measurements.analysis.utils import parse_pose_estimation, calculate_euclidean_distance, calculate_real_width, extract_identifier_from_gt, calculate_bbox_area
import math
import re


"""
FiftyOne data loader for prawn measurement validation.

This module handles:
1. Loading and processing YOLO keypoint detections
2. Creating FiftyOne visualizations
3. Calculating real-world measurements
4. Comparing predictions with ground truth

Image Specifications:
    - Width: 5312 pixels
    - Height: 2988 pixels
    - Camera: GoPro Hero 11
"""

class ObjectLengthMeasurer:
    def __init__(self, image_width, image_height, horizontal_fov, vertical_fov, distance_mm):
        self.image_width = image_width
        self.image_height = image_height
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.distance_mm = distance_mm
        self.scale_x, self.scale_y = self.calculate_scaling_factors()

    def calculate_scaling_factors(self):
        """
        Calculate the scaling factors (mm per pixel) based on the camera's FOV and distance.
        """
        fov_x_rad = math.radians(self.horizontal_fov)
        fov_y_rad = math.radians(self.vertical_fov)
        scale_x = (2 * self.distance_mm * math.tan(fov_x_rad / 2)) / self.image_width
        scale_y = (2 * self.distance_mm * math.tan(fov_y_rad / 2)) / self.image_height
        return scale_x, scale_y

    def normalize_angle(self, angle):
        """
        Normalize the angle to [0°, 90°].
        """
        theta_norm = min(abs(angle % 180), 180 - abs(angle % 180))
        return theta_norm

    def compute_length(self, predicted_length, angle_deg):
        """
        Compute the real-world length in millimeters using combined scaling factors.
        """
        angle_rad = math.radians(angle_deg)
        combined_scale = math.sqrt((self.scale_x * math.cos(angle_rad)) ** 2 + 
                                   (self.scale_y * math.sin(angle_rad)) ** 2)
        length_mm = predicted_length * combined_scale
        return length_mm, combined_scale

    def compute_length_two_points(self, point1_low_res, point2_low_res):
        """
        Compute the real-world distance between two points in the low-resolution image.
        """
        delta_x_low = point2_low_res[0] - point1_low_res[0]
        delta_y_low = point2_low_res[1] - point1_low_res[1]
        distance_px = math.sqrt(delta_x_low ** 2 + delta_y_low ** 2)
        
        angle_rad = math.atan2(delta_y_low, delta_x_low)
        angle_deg = math.degrees(angle_rad)
        normalized_angle = self.normalize_angle(angle_deg)
        
        distance_mm, combined_scale = self.compute_length(distance_px, normalized_angle)
        
        return distance_mm, combined_scale, normalized_angle, distance_px

def load_data(filtered_data_path, metadata_path):
    filtered_df = pd.read_csv(filtered_data_path)
    metadata_df = pd.read_excel(metadata_path)
    return filtered_df, metadata_df

def load_data_body(filtered_data_path, metadata_path):
    filtered_df = pd.read_excel(filtered_data_path)
    metadata_df = pd.read_excel(metadata_path)
    return filtered_df, metadata_df

def create_dataset(measurement_type, weights_type):
    if not os.path.exists(f"/Users/gilbenor/Documents/code_projects/msc/counting_research_algorithms/fiftyone_datasets/{measurement_type}_{weights_type}"):
        print(f"Dataset {measurement_type}_{weights_type} does not exist")
        dataset = fo.Dataset(f"prawn_dataset_{measurement_type}_{weights_type}", overwrite=True, persistent=True)
        dataset.default_skeleton = fo.KeypointSkeleton(
            labels=["start_carapace", "eyes", "rostrum", "tail"],
            edges=[
                [0, 1],  # start_carapace to eyes
                [1, 2],  # eyes to rostrum
                [0, 3]   # start_carapace to tail
            ]
        )
        return dataset, False
    else:
        try:
            dataset = fo.load_dataset(f"prawn_dataset_{measurement_type}_{weights_type}")
            if dataset:
                print(f"Dataset {measurement_type}_{weights_type} exists")
                return dataset, True
        except:
            print(f"Dataset {measurement_type}_{weights_type} does not exist")
            dataset = fo.Dataset(f"prawn_dataset_{measurement_type}_{weights_type}", overwrite=True, persistent=True)
            dataset.default_skeleton = fo.KeypointSkeleton(
                labels=["start_carapace", "eyes", "rostrum", "tail"],
                edges=[
                    [0, 1],  # start_carapace to eyes
                    [1, 2],  # eyes to rostrum
                    [0, 3]   # start_carapace to tail
                ]
            )
            return dataset, False

def create_dataset_body(measurement_type, weights_type):
    if not os.path.exists(f"/Users/gilbenor/Documents/code_projects/msc/counting_research_algorithms/fiftyone_datasets/{measurement_type}_{weights_type}"):
        print(f"Dataset {measurement_type}_{weights_type} does not exist")
        dataset = fo.Dataset(f"prawn_dataset_{measurement_type}_{weights_type}", overwrite=True, persistent=True)
        dataset.default_skeleton = fo.KeypointSkeleton(
            labels=["start_carapace", "eyes", "rostrum", "tail"],
            edges=[
                [0, 1],  # start_carapace to eyes
                [1, 2],  # eyes to rostrum
                [0, 3]   # start_carapace to tail
            ]
        )
        return dataset, False

    try:
        dataset = fo.load_dataset(f"prawn_dataset_{measurement_type}_{weights_type}")
        if dataset:
            print(f"Dataset {measurement_type}_{weights_type} exists")
            return dataset, True
    except:
        dataset = fo.Dataset(f"prawn_dataset_{measurement_type}_{weights_type}", overwrite=True, persistent=True)
        dataset.default_skeleton = fo.KeypointSkeleton(
            labels=["start_carapace", "eyes", "rostrum", "tail"],
            edges=[
                [0, 1],  # start_carapace to eyes
                [1, 2],  # eyes to rostrum
                [0, 3]   # start_carapace to tail
            ]
        )
        return dataset, False

def process_poses(poses, is_ground_truth=False):
    """
    Process YOLO keypoint detections into FiftyOne format with enhanced visibility checking.
    """
    keypoints_list = []
    detections = []

    # Constants for visibility checking
    EDGE_THRESHOLD = 0.01  # Points within 1% of image edge are invalid
    CONFIDENCE_THRESHOLD = 0.3  # Points with confidence below 30% are invalid
    MIN_DISTANCE = 5  # Minimum distance between points in pixels

    for pose in poses:
        if len(pose) == 17:  # 1 class + 4 bbox + 4 keypoints × 3 values
            # Bounding box processing
            x1_rel, y1_rel, width_rel, height_rel = pose[1:5]
            x1_rel -= width_rel / 2
            y1_rel -= height_rel / 2

            # Process keypoints with visibility checks
            keypoints = []
            confidences = []
            visibility_flags = []

            for i in range(5, len(pose), 3):
                x, y, conf = pose[i:i+3]
                
                # Check visibility conditions
                is_visible = True
                visibility_reason = []

                # Edge check
                if x < EDGE_THRESHOLD or x > (1 - EDGE_THRESHOLD) or y < EDGE_THRESHOLD or y > (1 - EDGE_THRESHOLD):
                    is_visible = False
                    visibility_reason.append("edge")

                # Confidence check
                if conf < CONFIDENCE_THRESHOLD:
                    is_visible = False
                    visibility_reason.append("confidence")

                # Distance check with previous points
                if keypoints:
                    min_dist = float('inf')
                    for prev_x, prev_y in keypoints:
                        dist = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                        min_dist = min(min_dist, dist)
                    if min_dist < MIN_DISTANCE:
                        is_visible = False
                        visibility_reason.append("distance")

                keypoints.append([x, y])
                confidences.append(conf)
                visibility_flags.append({
                    'visible': is_visible,
                    'reason': visibility_reason if not is_visible else None
                })

            # Create keypoint object with visibility information
            keypoint = fo.Keypoint(
                points=keypoints,
                confidence=confidences,
                visibility=visibility_flags
            )
            keypoints_list.append(keypoint)

            # Create keypoints dictionary
            keypoints_dict = {
                'start_carapace': {'point': keypoints[0], 'conf': confidences[0], 'visibility': visibility_flags[0]},
                'eyes': {'point': keypoints[1], 'conf': confidences[1], 'visibility': visibility_flags[1]},
                'rostrum': {'point': keypoints[2], 'conf': confidences[2], 'visibility': visibility_flags[2]},
                'tail': {'point': keypoints[3], 'conf': confidences[3], 'visibility': visibility_flags[3]},
                'keypoint_ID': keypoint.id
            }

            # Count valid keypoints
            valid_keypoints = sum(1 for flag in visibility_flags if flag['visible'])

            # Create detection with visibility information
            detection_label = "prawn_truth" if is_ground_truth else "prawn"
            detections.append(fo.Detection(
                label=detection_label,
                bounding_box=[x1_rel, y1_rel, width_rel, height_rel],
                attributes={
                    'keypoints': keypoints_dict,
                    'valid_keypoints': valid_keypoints,
                    'total_keypoints': len(keypoints)
                }
            ))
    
    return keypoints_list, detections

def process_images(image_paths, prediction_folder_path, ground_truth_paths_text, filtered_df, metadata_df, dataset, pond_type, measurement_type):
    """
    Process images and add them to the FiftyOne dataset.
    """
    for image_path in tqdm(image_paths, desc="Processing images"):
        filename = os.path.basename(image_path)
        sample = fo.Sample(filepath=image_path)
        sample.tags = [pond_type]

        # Load ground truth
        ground_truth_path = os.path.join(ground_truth_paths_text, filename.replace('.jpg', '.txt'))
        if os.path.exists(ground_truth_path):
            ground_truth_poses = parse_pose_estimation(ground_truth_path)
            keypoints_list, detections = process_poses(ground_truth_poses, is_ground_truth=True)
            sample["ground_truth"] = fo.Detections(detections=detections)
            sample["keypoints_ground_truth"] = fo.Keypoints(keypoints=keypoints_list)

        # Load predictions
        prediction_path = os.path.join(prediction_folder_path, filename.replace('.jpg', '.txt'))
        if os.path.exists(prediction_path):
            prediction_poses = parse_pose_estimation(prediction_path)
            keypoints_list, detections = process_poses(prediction_poses)
            sample["detections_predictions"] = fo.Detections(detections=detections)
            sample["keypoints_predictions"] = fo.Keypoints(keypoints=keypoints_list)

        # Add metadata and process detections
        if measurement_type == 'carapace':
            add_metadata(sample, filename, filtered_df, metadata_df)
        else:
            add_metadata_body(sample, filename, filtered_df, metadata_df)

        dataset.add_sample(sample)

def process_images_body(image_paths, prediction_txt_path, ground_truth_paths_text, filtered_df, metadata_df, dataset, pond_type):
    """
    Process images for body measurements and add them to the FiftyOne dataset.
    """
    for image_path in tqdm(image_paths, desc="Processing images"):
        filename = os.path.basename(image_path)
        sample = fo.Sample(filepath=image_path)
        sample.tags = [pond_type]

        # Load ground truth
        ground_truth_path = os.path.join(ground_truth_paths_text, filename.replace('.jpg', '.txt'))
        if os.path.exists(ground_truth_path):
            ground_truth_poses = parse_pose_estimation(ground_truth_path)
            keypoints_list, detections = process_poses(ground_truth_poses, is_ground_truth=True)
            sample["ground_truth"] = fo.Detections(detections=detections)
            sample["keypoints_ground_truth"] = fo.Keypoints(keypoints=keypoints_list)

        # Load predictions
        prediction_path = os.path.join(prediction_txt_path, filename.replace('.jpg', '.txt'))
        if os.path.exists(prediction_path):
            prediction_poses = parse_pose_estimation(prediction_path)
            keypoints_list, detections = process_poses(prediction_poses)
            sample["detections_predictions"] = fo.Detections(detections=detections)
            sample["keypoints_predictions"] = fo.Keypoints(keypoints=keypoints_list)

        # Add metadata and process detections
        add_metadata_body(sample, filename, filtered_df, metadata_df)
        dataset.add_sample(sample)

def extract_identifier(filename):
    """
    Extract identifier pattern like 'GX010179_200_3927' from filename.
    """
    pattern = r'(GX\\d+_\\d+_\\d+)'
    match = re.search(pattern, filename)
    return match.group(1) if match else None

def add_metadata(sample, filename, filtered_df, metadata_df, swimmingdf=None):
    """
    Add metadata from Excel file to FiftyOne sample and process detections.
    """
    print(filename)
    if 'undistorted' in filename:
        filename = filename.replace('undistorted_', '')
    filename = filename.split('.')[0]
    compatible_file_name = filename.split('_')[0:3]

    matching_rows = filtered_df[filtered_df['Label'].str.contains('_'.join(compatible_file_name))]
    filename = matching_rows['Label'].values[0].split(':')[1] 

    joined_string = '_'.join([compatible_file_name[0], compatible_file_name[-1]])
    relevant_part = joined_string 
    
    metadata_row = metadata_df[metadata_df['file_name_new'] == relevant_part]
    if not metadata_row.empty:
        metadata = metadata_row.iloc[0].to_dict() 
        for key, value in metadata.items():
            if key != 'file name':
                sample[key] = value
    else:
        print(f"No metadata found for {relevant_part}")
    
    add_prawn_detections(sample, matching_rows, filtered_df, filename)

def add_metadata_body(sample, filename, filtered_df, metadata_df, swimmingdf=None):
    """
    Add metadata from Excel file to FiftyOne sample and process detections.
    """
    identifier = extract_identifier(filename)
    print(f'identifier {identifier}')

    matching_rows = pd.DataFrame()
    for index, row in filtered_df.iterrows():
        if identifier in str(row['Label']):
            matching_rows = filtered_df.loc[filtered_df['Label'] == row['Label']]
            break
    if matching_rows.empty:
        print(f'no matching rows found for {identifier}')
        return
    
    relevant_part = identifier.split('_')[0] + '_' + identifier.split('_')[-1]
    metadata_row = metadata_df[metadata_df['file_name_new'] == relevant_part]
    if not metadata_row.empty:
        metadata = metadata_row.iloc[0].to_dict() 
        for key, value in metadata.items():
            if key != 'file name':
                sample[key] = value
    else:
        print(f"No metadata found for {relevant_part}")
    
    for file in filtered_df['Label'].unique():
        if identifier in file.split(':')[1]:
            filename = file.split(':')[1]
            break

    add_prawn_detections_body(sample, matching_rows, filtered_df, filename)

def find_closest_detection(sample, prawn_bbox):
    """
    Find closest YOLO detection to a ground truth bounding box.
    """
    prawn_point = (prawn_bbox[0] / 5312, prawn_bbox[1] / 2988)
    min_distance = float('inf')
    closest_detection_pred = None
    closest_detection_ground_truth = None
    
    for detection_bbox in sample["detections_predictions"].detections:
        det_point = (detection_bbox.bounding_box[0], detection_bbox.bounding_box[1])
        distance = calculate_euclidean_distance(prawn_point, det_point)
        if distance < min_distance:
            min_distance = distance
            closest_detection_pred = detection_bbox

    min_distance = float('inf')
    for detection_bbox_ground_truth in sample["ground_truth"].detections:
        det_point = (detection_bbox_ground_truth.bounding_box[0], detection_bbox_ground_truth.bounding_box[1])
        distance = calculate_euclidean_distance(prawn_point, det_point)
        if distance < min_distance:
            min_distance = distance
            closest_detection_ground_truth = detection_bbox_ground_truth
    
    return closest_detection_pred, closest_detection_ground_truth

def add_prawn_detections(sample, matching_rows, filtered_df, filename):
    """
    Add prawn detections and visualizations to a FiftyOne sample.
    """
    min_diagonal_line_1 = []
    min_diagonal_line_2 = []
    max_diagonal_line_1 = []
    max_diagonal_line_2 = []
    mid_diagonal_line_1 = []
    mid_diagonal_line_2 = []

    for _, row in matching_rows.iterrows():
        prawn_id = row['PrawnID']
        bounding_boxes = []
        
        length_1 = abs(float(row['Length_1'])) if pd.notna(row['Length_1']) else None
        length_2 = abs(float(row['Length_2'])) if pd.notna(row['Length_2']) else None
        length_3 = abs(float(row['Length_3'])) if pd.notna(row['Length_3']) else None
        
        for bbox_key, length in zip(['BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3'],
                                [length_1, length_2, length_3]):
            if pd.notna(row[bbox_key]) and length is not None:
                bbox = ast.literal_eval(row[bbox_key])
                bbox = tuple(float(coord) for coord in bbox)
                bounding_boxes.append((bbox, length))
        
        if not bounding_boxes:
            print(f"No bounding boxes found for prawn ID {prawn_id} in {filename}.")
            continue

        sorted_boxes = sorted(bounding_boxes, key=lambda x: calculate_bbox_area(x[0]))
        min_bbox, min_length = sorted_boxes[0]
        mid_bbox, mid_length = sorted_boxes[1]
        max_bbox, max_length = sorted_boxes[2]
        
        prawn_min_normalized_bbox = [min_bbox[0] / 5312, min_bbox[1] / 2988, min_bbox[2] / 5312, min_bbox[3] / 2988]
        prawn_mid_normalized_bbox = [mid_bbox[0] / 5312, mid_bbox[1] / 2988, mid_bbox[2] / 5312, mid_bbox[3] / 2988]
        prawn_max_normalized_bbox = [max_bbox[0] / 5312, max_bbox[1] / 2988, max_bbox[2] / 5312, max_bbox[3] / 2988]

        closest_detection, ground = find_closest_detection(sample, min_bbox)

        if closest_detection is not None:
            process_detection(closest_detection, sample, filename, prawn_id, filtered_df, ground)

        # Add diagonal lines for visualization
        for bbox, length, color_pair in [
            (prawn_max_normalized_bbox, max_length, ("red", "yellow")),
            (prawn_mid_normalized_bbox, mid_length, ("blue", "green")),
            (prawn_min_normalized_bbox, min_length, ("blue", "green"))
        ]:
            x_min, y_min, width, height = bbox
            
            top_left = [x_min, y_min]
            top_right = [x_min + width, y_min]
            bottom_left = [x_min, y_min + height]
            bottom_right = [x_min + width, y_min + height]

            diagonal1 = [top_left, bottom_right]
            diagonal2 = [top_right, bottom_left]

            for points, color, line_list in [
                (diagonal1, color_pair[0], [min_diagonal_line_1, mid_diagonal_line_1, max_diagonal_line_1][0]),
                (diagonal2, color_pair[1], [min_diagonal_line_2, mid_diagonal_line_2, max_diagonal_line_2][0])
            ]:
                polyline = fo.Polyline(
                    label=f"{length:.2f}mm",
                    points=[points],
                    closed=False,
                    filled=False,
                    line_color=color,
                    thickness=2
                )
                line_list.append(polyline)

        # Add all diagonal lines to sample
        sample["min_diagonal_line_1"] = fo.Polylines(polylines=min_diagonal_line_1)
        sample["min_diagonal_line_2"] = fo.Polylines(polylines=min_diagonal_line_2)
        sample["max_diagonal_line_1"] = fo.Polylines(polylines=max_diagonal_line_1)
        sample["max_diagonal_line_2"] = fo.Polylines(polylines=max_diagonal_line_2)
        sample["mid_diagonal_line_1"] = fo.Polylines(polylines=mid_diagonal_line_1)
        sample["mid_diagonal_line_2"] = fo.Polylines(polylines=mid_diagonal_line_2)

def add_prawn_detections_body(sample, matching_rows, filtered_df, filename):
    """
    Add prawn detections and visualizations to a FiftyOne sample for body measurements.
    """
    min_diagonal_line_1 = []
    min_diagonal_line_2 = []
    max_diagonal_line_1 = []
    max_diagonal_line_2 = []
    mid_diagonal_line_1 = []
    mid_diagonal_line_2 = []

    for _, row in matching_rows.iterrows():
        prawn_id = row['PrawnID']
        bounding_boxes = []
        
        length_1 = abs(float(row['Length_1'])) if pd.notna(row['Length_1']) else None
        length_2 = abs(float(row['Length_2'])) if pd.notna(row['Length_2']) else None
        length_3 = abs(float(row['Length_3'])) if pd.notna(row['Length_3']) else None
        
        for bbox_key, length in zip(['BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3'],
                                [length_1, length_2, length_3]):
            if pd.notna(row[bbox_key]) and length is not None:
                bbox = ast.literal_eval(row[bbox_key])
                bbox = tuple(float(coord) for coord in bbox)
                bounding_boxes.append((bbox, length))
        
        if not bounding_boxes:
            print(f"No bounding boxes found for prawn ID {prawn_id} in {filename}.")
            continue

        sorted_boxes = sorted(bounding_boxes, key=lambda x: calculate_bbox_area(x[0]))
        min_bbox, min_length = sorted_boxes[0]
        mid_bbox, mid_length = sorted_boxes[1]
        max_bbox, max_length = sorted_boxes[2]

        closest_detection, ground = find_closest_detection(sample, min_bbox)

        if closest_detection is not None:
            process_detection_body(closest_detection, sample, filename, prawn_id, filtered_df, ground)

        # Add visualization lines similar to add_prawn_detections
        # ... (rest of visualization code similar to add_prawn_detections)

def process_detection(closest_detection, sample, filename, prawn_id, filtered_df, ground):
    """
    Process matched detections and calculate measurements.
    """
    height_mm = sample['height(mm)']
    if sample.tags[0] == 'test-left' or sample.tags[0] == 'test-right':
        focal_length = 23.64
    else:
        focal_length = 24.72

    pixel_size = 0.00716844  # Pixel size in mm

    keypoints_dict = closest_detection.attributes["keypoints"]
    carapace_points = [keypoints_dict['start_carapace'], keypoints_dict['eyes']]

    if any(math.isnan(coord) for point in carapace_points for coord in point):
        print(f"Warning: Keypoint at edge for prawn {prawn_id} in {filename}")
        return

    keypoint1_scaled = [carapace_points[0][0] * 5312, carapace_points[0][1] * 2988]
    keypoint2_scaled = [carapace_points[1][0] * 5312, carapace_points[1][1] * 2988]

    euclidean_distance_pixels = calculate_euclidean_distance(keypoint1_scaled, keypoint2_scaled)
    focal_real_length_cm = calculate_real_width(focal_length, height_mm, euclidean_distance_pixels, pixel_size)

    object_length_measurer = ObjectLengthMeasurer(5312, 2988, 76.2, 46, height_mm)
    distance_mm, combined_scale, angle_deg, distance_px = object_length_measurer.compute_length_two_points(keypoint1_scaled, keypoint2_scaled)

    keypoints_dict_ground = ground.attributes["keypoints"]
    keypoints_ground = [keypoints_dict_ground['start_carapace'], keypoints_dict_ground['eyes']]
    keypoint1_scaled_ground = [keypoints_ground[0][0] * 5312, keypoints_ground[0][1] * 2988]
    keypoint2_scaled_ground = [keypoints_ground[1][0] * 5312, keypoints_ground[1][1] * 2988]

    object_length_measurer_ground = ObjectLengthMeasurer(5312, 2988, 76.2, 46, height_mm)
    distance_mm_ground, combined_scale_ground, angle_deg_ground, distance_px_ground = object_length_measurer_ground.compute_length_two_points(keypoint1_scaled_ground, keypoint2_scaled_ground)

    # Update filtered_df with measurements
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_ground_truth_annotation(mm)'] = distance_mm_ground
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_ground_truth_annotation_pixels'] = distance_px_ground
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'pred_Distance_pixels'] = distance_px
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'combined_scale'] = combined_scale
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'combined_scale_ground'] = combined_scale_ground
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'id'] = keypoint_id
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_fov(mm)'] = distance_mm
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Height(mm)'] = height_mm
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'focal_RealLength(cm)'] = focal_real_length_cm

    print(f'{filename}  {prawn_id} ')

    # Calculate error metrics
    min_true_length = min(abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),
                         abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),
                         abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))

    max_true_length = max(abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),
                         abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),
                         abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))

    median_true_length = sorted([
        abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),
        abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),
        abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0])
    ])[1]

    # Calculate and store error percentages
    error_percentage_min = abs(distance_mm - min_true_length) / min_true_length * 100
    error_percentage_max = abs(distance_mm - max_true_length) / max_true_length * 100
    error_percentage_median = abs(distance_mm - median_true_length) / median_true_length * 100

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_min'] = error_percentage_min
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_max'] = error_percentage_max
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_median'] = error_percentage_median

    # Update detection labels
    ground_truth_detection_label = f'prawn_truth{distance_mm_ground:.2f}mm'
    ground.label = ground_truth_detection_label

    closest_detection_label = f'pred_length: {distance_mm:.2f}mm'
    closest_detection.label = closest_detection_label
    closest_detection.attributes["prawn_id"] = fo.Attribute(value=prawn_id)

    # Add error tags to sample
    min_error_percentage = min(error_percentage_min, error_percentage_max, error_percentage_median)
    
    if min_error_percentage > 50:
        if "MPE_fov>50" not in sample.tags:
            sample.tags.append("MPE_fov>50")
    elif min_error_percentage > 25:
        if "MPE_fov>25" not in sample.tags:
            sample.tags.append("MPE_fov>25")
    elif min_error_percentage > 10:
        if "MPE_fov>10" not in sample.tags:
            sample.tags.append("MPE_fov>10")
    elif min_error_percentage > 5:
        if "MPE_fov>5" not in sample.tags:
            sample.tags.append("MPE_fov>5")
    else:
        if "MPE_fov<5" not in sample.tags:
            sample.tags.append("MPE_fov<5")

def process_detection_body(closest_detection, sample, filename, prawn_id, filtered_df, ground):
    """
    Process matched detections and calculate measurements for body measurements.
    """
    height_mm = sample['height(mm)']
    if sample.tags[0] == 'test-left' or sample.tags[0] == 'test-right':
        focal_length = 23.64
    else:
        focal_length = 24.72

    pixel_size = 0.00716844

    keypoints_dict = closest_detection.attributes["keypoints"]
    total_length_points = [keypoints_dict['tail'], keypoints_dict['rostrum']]

    if any(math.isnan(coord) for point in total_length_points for coord in point):
        print(f"Warning: Keypoint at edge for prawn {prawn_id} in {filename}")
        return

    keypoint1_scaled = [total_length_points[0][0] * 5312, total_length_points[0][1] * 2988]
    keypoint2_scaled = [total_length_points[1][0] * 5312, total_length_points[1][1] * 2988]

    euclidean_distance_pixels = calculate_euclidean_distance(keypoint1_scaled, keypoint2_scaled)
    focal_real_length_cm = calculate_real_width(focal_length, height_mm, euclidean_distance_pixels, pixel_size)

    object_length_measurer = ObjectLengthMeasurer(5312, 2988, 75.2, 46, height_mm)
    distance_mm, combined_scale, angle_deg, distance_px = object_length_measurer.compute_length_two_points(keypoint1_scaled, keypoint2_scaled)

    keypoints_dict_ground = ground.attributes["keypoints"]
    keypoints_ground = [keypoints_dict_ground['tail'], keypoints_dict_ground['rostrum']]
    keypoint1_scaled_ground = [keypoints_ground[0][0] * 5312, keypoints_ground[0][1] * 2988]
    keypoint2_scaled_ground = [keypoints_ground[1][0] * 5312, keypoints_ground[1][1] * 2988]

    object_length_measurer_ground = ObjectLengthMeasurer(5312, 2988, 75.2, 46, height_mm)
    distance_mm_ground, combined_scale_ground, angle_deg_ground, distance_px_ground = object_length_measurer_ground.compute_length_two_points(keypoint1_scaled_ground, keypoint2_scaled_ground)

    # Update measurements in filtered_df
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_ground_truth_annotation(mm)'] = distance_mm_ground
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_ground_truth_annotation_pixels'] = distance_px_ground
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'pred_Distance_pixels'] = distance_px
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'combined_scale'] = combined_scale
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'combined_scale_ground'] = combined_scale_ground
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'id'] = keypoint_id
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_fov(mm)'] = distance_mm
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Height(mm)'] = height_mm
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'focal_RealLength(cm)'] = focal_real_length_cm
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Pond_Type'] = sample.tags[0]
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Euclidean_Distance'] = euclidean_distance_pixels

    print(f'{filename}  {prawn_id} ')

    # Calculate true lengths and errors
    min_true_length = min(abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),
                         abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),
                         abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))

    max_true_length = max(abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),
                         abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),
                         abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))

    median_true_length = sorted([
        abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),
        abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),
        abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0])
    ])[1]

    # Calculate and store error metrics
    error_percentage_min = abs(distance_mm - min_true_length) / min_true_length * 100
    error_percentage_max = abs(distance_mm - max_true_length) / max_true_length * 100
    error_percentage_median = abs(distance_mm - median_true_length) / median_true_length * 100

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_min'] = error_percentage_min
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_max'] = error_percentage_max
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_median'] = error_percentage_median

    # Update detection labels
    ground_truth_detection_label = f'prawn_truth{distance_mm_ground:.2f}mm'
    ground.label = ground_truth_detection_label

    closest_detection_label = f'pred_length: {distance_mm:.2f}mm'
    closest_detection.label = closest_detection_label
    closest_detection.attributes["prawn_id"] = fo.Attribute(value=prawn_id)

    # Add error tags
    min_error_percentage = min(error_percentage_min, error_percentage_max, error_percentage_median)
    
    if min_error_percentage > 50:
        if "MPE_fov>50" not in sample.tags:
            sample.tags.append("MPE_fov>50")
    elif min_error_percentage > 25:
        if "MPE_fov>25" not in sample.tags:
            sample.tags.append("MPE_fov>25")
    elif min_error_percentage > 10:
        if "MPE_fov>10" not in sample.tags:
            sample.tags.append("MPE_fov>10")
    elif min_error_percentage > 5:
        if "MPE_fov>5" not in sample.tags:
            sample.tags.append("MPE_fov>5")
    else:
        if "MPE_fov<5" not in sample.tags:
            sample.tags.append("MPE_fov<5")
