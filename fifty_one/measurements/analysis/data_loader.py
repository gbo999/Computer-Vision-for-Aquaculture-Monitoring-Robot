# data_loader.py
import fiftyone as fo
import pandas as pd
import os
import ast
from tqdm import tqdm
from utils import parse_pose_estimation, calculate_euclidean_distance, calculate_real_width, extract_identifier_from_gt, calculate_bbox_area
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
        # self.to_scale_x = image_width / 640  # Assuming low-res width is 640
        # self.to_scale_y = image_height / 360  # Assuming low-res height is 360

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
        if angle < 0:
            angle += 90
        return abs(angle)

    def compute_length(self, predicted_length, angle_deg):
        """
        Compute the real-world length in millimeters using combined scaling factors.
        """
        angle_rad = math.radians(angle_deg)
        combined_scale = math.sqrt((self.scale_x * math.cos(angle_rad)) ** 2 + 
                                   (self.scale_y * math.sin(angle_rad)) ** 2)
        length_mm = predicted_length * combined_scale
        return length_mm

    def compute_length_two_points(self, point1_low_res, point2_low_res):
        """
        Compute the real-world distance between two points in the low-resolution image.
        
        Parameters:
        - point1_low_res: Tuple (x1, y1) coordinates of the first point in low-res pixels.
        - point2_low_res: Tuple (x2, y2) coordinates of the second point in low-res pixels.
        
        Returns:
        - distance_mm: Real-world distance between the two points in millimeters.
        - angle_deg: Angle of the line connecting the two points relative to the horizontal axis in degrees.
        """
        # Calculate pixel distance in low-res image
        delta_x_low = point2_low_res[0] - point1_low_res[0]
        delta_y_low = point2_low_res[1] - point1_low_res[1]
        distance_px = math.sqrt(delta_x_low ** 2 + delta_y_low ** 2)
        




        # Calculate angle in degrees
        angle_rad = math.atan2(delta_y_low, delta_x_low)
        angle_deg = math.degrees(angle_rad)
        normalized_angle = self.normalize_angle(angle_deg)
        
        # Scale the pixel distance from low-res to high-res
        # distance_px_high = distance_px_low * self.to_scale_x  # Assuming uniform scaling
        
        # Compute real-world distance
        distance_mm = self.compute_length(distance_px, normalized_angle)
        
        return distance_mm, normalized_angle, distance_px

def load_data(filtered_data_path, metadata_path):
    filtered_df = pd.read_csv(filtered_data_path)
    metadata_df = pd.read_excel(metadata_path)
    return filtered_df, metadata_df


def load_data_body(filtered_data_path, metadata_path):
    filtered_df = pd.read_excel(filtered_data_path)
    metadata_df = pd.read_excel(metadata_path)
    return filtered_df, metadata_df


def create_dataset(measurement_type,weights_type):

    if not os.path.exists(f"/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/thesisi/thesis document/{measurement_type}_{weights_type}"):
        print(f"Dataset {measurement_type}_{weights_type} does not exist")
        dataset = fo.Dataset(f"prawn_dataset_{measurement_type}_{weights_type}", overwrite=True, persistent=True)
        dataset.default_skeleton = fo.KeypointSkeleton(
                labels=["start_carapace", "eyes", "rostrum", "tail"],  # Match YOLO order
                edges=[
                    [0, 1],  # start_carapace to eyes
                    [1, 2],  # eyes to rostrum
                    [0, 3]   # start_carapace to tail
                ]
            )
        return dataset,False

    else:
        try:
            dataset = fo.load_dataset(f"prawn_dataset_{measurement_type}_{weights_type}")
            if dataset:
                print(f"Dataset {measurement_type}_{weights_type} exists")
                return dataset,True
        except:
            print(f"Dataset {measurement_type}_{weights_type} does not exist")
            dataset = fo.Dataset(f"prawn_dataset_{measurement_type}_{weights_type}", overwrite=True, persistent=True)
            dataset.default_skeleton = fo.KeypointSkeleton(
                    labels=["start_carapace", "eyes", "rostrum", "tail"],  # Match YOLO order
                    edges=[
                        [0, 1],  # start_carapace to eyes
                        [1, 2],  # eyes to rostrum
                        [0, 3]   # start_carapace to tail
                    ]
                )
            return dataset,False
            

def create_dataset_body(measurement_type,weights_type):
    #dataset exists
    if not os.path.exists(f"/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/thesisi/thesis document/{measurement_type}_{weights_type}"):
        print(f"Dataset {measurement_type}_{weights_type} does not exist")
        dataset = fo.Dataset(f"prawn_dataset_{measurement_type}_{weights_type}", overwrite=True, persistent=True)
        dataset.default_skeleton = fo.KeypointSkeleton(
                labels=["start_carapace", "eyes", "rostrum", "tail"],  # Match YOLO order
                edges=[
                    [0, 1],  # start_carapace to eyes
                    [1, 2],  # eyes to rostrum
                    [0, 3]   # start_carapace to tail
                ]
            )
        return dataset,False

    try:
        dataset = fo.load_dataset(f"prawn_dataset_{measurement_type}_{weights_type}")
        if dataset:
            print(f"Dataset {measurement_type}_{weights_type} exists")
            return dataset,True
    except:
        dataset = fo.Dataset(f"prawn_dataset_{measurement_type}_{weights_type}", overwrite=True, persistent=True)
        dataset.default_skeleton = fo.KeypointSkeleton(
            labels=["start_carapace", "eyes", "rostrum", "tail"],  # Match YOLO order
            edges=[
                [0, 1],  # start_carapace to eyes
                [1, 2],  # eyes to rostrum
                [0, 3]   # start_carapace to tail
            ]
        )
        return dataset,False

def process_poses(poses, is_ground_truth=False):
    """
    Process YOLO keypoint detections into FiftyOne format.

    Args:
        poses (list): List of YOLO pose detections
            Format: [class_id, x, y, w, h, kp1_x, kp1_y, kp1_conf, kp2_x, kp2_y, kp2_conf, ...]
        is_ground_truth (bool): Whether these are ground truth annotations

    Returns:
        tuple: (keypoints_list, detections)
            - keypoints_list: List of FiftyOne Keypoint objects
            - detections: List of FiftyOne Detection objects

    Keypoint order in YOLO format:
        0: start_carapace
        1: eyes
        2: rostrum
        3: tail
    """
    keypoints_list = []
    detections = []

    for pose in poses:
        if len(pose) == 17:  # 1 class + 4 bbox + 4 keypoints × 3 values
            # Bounding box processing
            x1_rel, y1_rel, width_rel, height_rel = pose[1:5]
            x1_rel -= width_rel / 2
            y1_rel -= height_rel / 2

            # Process keypoints
            keypoints = [[pose[i], pose[i + 1]] for i in range(5, len(pose), 3)]
            keypoint = fo.Keypoint(points=keypoints)
            keypoints_list.append(keypoint)

            # Create keypoints dictionary with correct order
            keypoints_dict = {
                'start_carapace': keypoints[0],  # Index 0 in YOLO format
                'eyes': keypoints[1],            # Index 1 in YOLO format
                'rostrum': keypoints[2],         # Index 2 in YOLO format
                'tail': keypoints[3],            # Index 3 in YOLO format
                'keypoint_ID': keypoint.id
            }

            # Create detection
            if not is_ground_truth:
                detections.append(fo.Detection(
                    label="prawn",
                    bounding_box=[x1_rel, y1_rel, width_rel, height_rel],
                    attributes={'keypoints': keypoints_dict}
                ))
            else:
                detections.append(fo.Detection(
                    label=f"prawn_truth",
                    bounding_box=[x1_rel, y1_rel, width_rel, height_rel],
                    attributes={'keypoints': keypoints_dict}
                ))
    
    return keypoints_list, detections





def add_metadata(sample, filename, filtered_df, metadata_df, swimmingdf=None):
    """
    Add metadata from Excel file to FiftyOne sample and process detections.

    This function:
    1. Processes filename to match metadata format
    2. Finds matching metadata in filtered_df and metadata_df
    3. Adds camera parameters and setup information to sample
    4. Triggers detection processing

    Args:
        sample (fo.Sample): FiftyOne sample to add metadata to
        filename (str): Image filename (e.g., 'GX010152_36_378.jpg_gamma')
        filtered_df (pd.DataFrame): DataFrame containing manual measurements and annotations
        metadata_df (pd.DataFrame): DataFrame containing camera setup parameters
        swimmingdf (pd.DataFrame, optional): Additional swimming data if available

    Example:
        Input filename: 'undistorted_GX010152_36_378.jpg_gamma'
        Compatible format: 'GX010152_36'
        Metadata matching: Uses 'GX010152_36' to find camera height, FOV, etc.

    Note:
        - Handles 'undistorted_' prefix in filenames
        - Splits filename to match metadata format
        - Adds all metadata except 'file name' to sample
        - Calls add_prawn_detections() for further processing
    """

    #undistorted_GX010073_55_1014.jpg_gamma
    print(filename)
    # Remove 'undistorted_' prefix if present
    if 'undistorted' in filename:
        filename = filename.replace('undistorted_', '')
    filename=filename.split('.')[0]
    # Extract compatible filename parts
    compatible_file_name = filename.split('_')[0:3]
    # print(compatible_file_name)


    # comp = compatible_file_name[1].split('-')[0]
    # compatible_file_name[2] = comp

    # print(f'compatible {compatible_file_name}')

    # Find matching rows in filtered_df
    matching_rows = filtered_df[filtered_df['Label'].str.contains('_'.join(compatible_file_name))]
    filename = matching_rows['Label'].values[0].split(':')[1] 

    # Create metadata lookup key
    joined_string = '_'.join([compatible_file_name[0], compatible_file_name[-1]])
    relevant_part = joined_string 
    
    # Find and add metadata
    metadata_row = metadata_df[metadata_df['file_name_new'] == relevant_part]
    if not metadata_row.empty:
        metadata = metadata_row.iloc[0].to_dict() 
        for key, value in metadata.items():
            if key != 'file name':
                sample[key] = value
    else:
        print(f"No metadata found for {relevant_part}")
    
    # Process detections
    add_prawn_detections(sample, matching_rows, filtered_df, filename)

    
    # add_prawn_detections_body(sample, matching_rows, filtered_df, filename)

def add_metadata_body(sample, filename, filtered_df, metadata_df, swimmingdf=None):
    """
    Add metadata from Excel file to FiftyOne sample and process detections.

    Args:
        sample (fo.Sample): FiftyOne sample to add metadata to
        filename (str): Image filename
        filtered_df (pd.DataFrame): DataFrame containing manual measurements
        metadata_df (pd.DataFrame): DataFrame containing camera setup parameters
        swimmingdf (pd.DataFrame, optional): Additional swimming data if available
    # """
    # print(f'filename {filename}')
    # # Remove 'undistorted_' prefix if present
    # if 'undistorted' in filename:
    #     filename = filename.replace('undistorted_', '')
    # filename=filename.split('.')[0]
    # # Extract compatible filename parts
    # compatible_file_name = filename.split('_')[0:3]
    # # print(compatible_file_name)
    # print(f'compatible_file_name {compatible_file_name}')



    identifier = extract_identifier(filename)
    print(f'identifier {identifier}')

    # comp = compatible_file_name[1].split('-')[0]
    # compatible_file_name[2] = comp

    # print(f'compatible {compatible_file_name}')

    matching_rows = pd.DataFrame()
    for index, row in filtered_df.iterrows():
        if identifier in str(row['Label']):
            matching_rows = filtered_df.loc[filtered_df['Label'] == row['Label']]
            break
    if matching_rows.empty:
        print(f'no matching rows found for {identifier}')
        return
    
    
    relevant_part =identifier.split('_')[0] + '_' + identifier.split('_')[-1]
    # Find and add metadata
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
            filename=file.split(':')[1]
            break



    # Process detections
    add_prawn_detections_body(sample, matching_rows, filtered_df, filename)

def add_prawn_detections_body(sample, matching_rows, filtered_df, filename):
    """
  Add prawn detections and visualizations to a FiftyOne sample.

    Args:
        sample (fo.Sample): FiftyOne sample to add detections to
        matching_rows (pd.DataFrame): Rows from filtered_df matching current image
        filtered_df (pd.DataFrame): DataFrame containing manual measurements
        filename (str): Current image filename

    Visualization Details:
        - Max bounding box: Red/Yellow diagonals
        - Min bounding box: Blue/Green diagonals
        - Each prawn gets 4 diagonal lines for size comparison

    Note:
        Bounding boxes are normalized to [0,1] range using:
        - x_normalized = x / 5312
        - y_normalized = y / 2988
    """
    # true_detections = []
    min_diagonal_line_1=[]
    min_diagonal_line_2=[]

    max_diagonal_line_1=[]
    max_diagonal_line_2=[]

    mid_diagonal_line_1=[]
    mid_diagonal_line_2=[]  




    for _, row in matching_rows.iterrows():
            prawn_id = row['PrawnID']
            bounding_boxes = []
            lengths = {}  # Store lengths with their corresponding bounding boxes
            
            # Get all lengths
            length_1 = abs(float(row['Length_1'])) if pd.notna(row['Length_1']) else None
            length_2 = abs(float(row['Length_2'])) if pd.notna(row['Length_2']) else None
            length_3 = abs(float(row['Length_3'])) if pd.notna(row['Length_3']) else None
            
            # Associate lengths with bounding boxes
            for bbox_key, length in zip(['BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3'],
                                    [length_1, length_2, length_3]):
                if pd.notna(row[bbox_key]) and length is not None:
                    bbox = ast.literal_eval(row[bbox_key])
                    bbox = tuple(float(coord) for coord in bbox)
                    bounding_boxes.append((bbox, length))
            
            if not bounding_boxes:
                print(f"No bounding boxes found for prawn ID {prawn_id} in {filename}.")
                continue

            # Sort bounding boxes by area and get their associated lengths
            sorted_boxes = sorted(bounding_boxes, key=lambda x: calculate_bbox_area(x[0]))
            min_bbox, min_length = sorted_boxes[0]
            mid_bbox, mid_length = sorted_boxes[1]
            max_bbox, max_length = sorted_boxes[2]
            

            prawn_min_normalized_bbox = [min_bbox[0] / 5312, min_bbox[1] / 2988, min_bbox[2] / 5312, min_bbox[3] / 2988]
            prawn_mid_normalized_bbox = [mid_bbox[0] / 5312, mid_bbox[1] / 2988, mid_bbox[2] / 5312, mid_bbox[3] / 2988]
            prawn_max_normalized_bbox = [max_bbox[0] / 5312, max_bbox[1] / 2988, max_bbox[2] / 5312, max_bbox[3] / 2988]

            closest_detection,ground = find_closest_detection(sample, min_bbox)

            if closest_detection is not None:
                process_detection_body(closest_detection, sample, filename, prawn_id, filtered_df,ground)

            x_min_max=prawn_max_normalized_bbox[0]
            y_min_max=prawn_max_normalized_bbox[1]
            width_max=prawn_max_normalized_bbox[2]
            heigh_maxt=prawn_max_normalized_bbox[3]

            # Corners in normalized coordinates
            top_left_max = [x_min_max, y_min_max]
            top_right_max = [x_min_max + width_max, y_min_max]
            bottom_left_max = [x_min_max, y_min_max + heigh_maxt]
            bottom_right_max = [x_min_max + width_max, y_min_max + heigh_maxt]

            # Diagonals
            diagonal1_max = [top_left_max, bottom_right_max]
            diagonal2_max = [top_right_max, bottom_left_max]

            diagonal1_polyline_max = fo.Polyline(
                label= "max diagonal- {} ".format(max_length),
                points=[diagonal1_max],
                closed=False,
                filled=False,
                line_color="red",
                thickness=2,
            )

            # Longest diagonal  
            diagonal2_polyline_max = fo.Polyline(
                label= "max diagonal- {} ".format(max_length),
                points=[diagonal2_max],
                closed=False,
                filled=False,
                line_color="yellow",
                thickness=2,
            )

            max_diagonal_line_1.append(diagonal1_polyline_max)
            max_diagonal_line_2.append(diagonal2_polyline_max)

            sample["max_diagonal_line_1"] = fo.Polylines(polylines=max_diagonal_line_1)
            sample["max_diagonal_line_2"] = fo.Polylines(polylines=max_diagonal_line_2)


            x_min_mid=prawn_mid_normalized_bbox[0]
            y_min_mid=prawn_mid_normalized_bbox[1]
            width_mid=prawn_mid_normalized_bbox[2]
            height_mid=prawn_mid_normalized_bbox[3]

            top_left_mid = [x_min_mid, y_min_mid]
            top_right_mid = [x_min_mid + width_mid, y_min_mid]
            bottom_left_mid = [x_min_mid, y_min_mid + height_mid]
            bottom_right_mid = [x_min_mid + width_mid, y_min_mid + height_mid]

            diagonal1_mid = [top_left_mid, bottom_right_mid]
            diagonal2_mid = [top_right_mid, bottom_left_mid]

            diagonal1_polyline_mid = fo.Polyline(
                label= "mid diagonal- {} ".format(mid_length),
                points=[diagonal1_mid],
                closed=False,
                filled=False,
                line_color="blue",
                thickness=2,
            )

            diagonal2_polyline_mid = fo.Polyline(
                label= "mid diagonal- {} ".format(mid_length),
                points=[diagonal2_mid],
                closed=False,
                filled=False,
                line_color="green",
                thickness=2,
            )

            mid_diagonal_line_1.append(diagonal1_polyline_mid)
            mid_diagonal_line_2.append(diagonal2_polyline_mid)

            sample["mid_diagonal_line_1"] = fo.Polylines(polylines=mid_diagonal_line_1)
            sample["mid_diagonal_line_2"] = fo.Polylines(polylines=mid_diagonal_line_2) 

            x_min_min=prawn_min_normalized_bbox[0]
            y_min_min=prawn_min_normalized_bbox[1]
            width_min=prawn_min_normalized_bbox[2]
            height_min=prawn_min_normalized_bbox[3]

            top_left_min = [x_min_min, y_min_min]
            top_right_min = [x_min_min + width_min, y_min_min]
            bottom_left_min = [x_min_min, y_min_min + height_min]
            bottom_right_min = [x_min_min + width_min, y_min_min + height_min]

            diagonal1_min = [top_left_min, bottom_right_min]
            diagonal2_min = [top_right_min, bottom_left_min]

            diagonal1_polyline_min = fo.Polyline(
                label= "min diagonal- {} ".format(min_length),
                points=[diagonal1_min],
                closed=False,
                filled=False,
                line_color="blue",
                thickness=2,
            )


            diagonal2_polyline_min = fo.Polyline(
                label= "min diagonal- {} ".format(min_length),
                points=[diagonal2_min],
                closed=False,
                filled=False,
                line_color="green",
                thickness=2,
            )

            min_diagonal_line_1.append(diagonal1_polyline_min)
            min_diagonal_line_2.append(diagonal2_polyline_min)

            sample["min_diagonal_line_1"] = fo.Polylines(polylines=min_diagonal_line_1)
            sample["min_diagonal_line_2"] = fo.Polylines(polylines=min_diagonal_line_2)








        


    



def add_prawn_detections(sample, matching_rows, filtered_df, filename):
    """
    Add prawn detections and visualizations to a FiftyOne sample.

    Args:
        sample (fo.Sample): FiftyOne sample to add detections to
        matching_rows (pd.DataFrame): Rows from filtered_df matching current image
        filtered_df (pd.DataFrame): DataFrame containing manual measurements
        filename (str): Current image filename

    Visualization Details:
        - Max bounding box: Red/Yellow diagonals
        - Min bounding box: Blue/Green diagonals
        - Each prawn gets 4 diagonal lines for size comparison

    Note:
        Bounding boxes are normalized to [0,1] range using:
        - x_normalized = x / 5312
        - y_normalized = y / 2988
    """
    # true_detections = []
    min_diagonal_line_1=[]
    min_diagonal_line_2=[]

    max_diagonal_line_1=[]
    max_diagonal_line_2=[]

    mid_diagonal_line_1=[]
    mid_diagonal_line_2=[]

    for _, row in matching_rows.iterrows():
            prawn_id = row['PrawnID']
            bounding_boxes = []
            lengths = {}  # Store lengths with their corresponding bounding boxes
            
            # Get all lengths
            length_1 = abs(float(row['Length_1'])) if pd.notna(row['Length_1']) else None
            length_2 = abs(float(row['Length_2'])) if pd.notna(row['Length_2']) else None
            length_3 = abs(float(row['Length_3'])) if pd.notna(row['Length_3']) else None
            
            # Associate lengths with bounding boxes
            for bbox_key, length in zip(['BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3'],
                                    [length_1, length_2, length_3]):
                if pd.notna(row[bbox_key]) and length is not None:
                    bbox = ast.literal_eval(row[bbox_key])
                    bbox = tuple(float(coord) for coord in bbox)
                    bounding_boxes.append((bbox, length))
            
            if not bounding_boxes:
                print(f"No bounding boxes found for prawn ID {prawn_id} in {filename}.")
                continue

            # Sort bounding boxes by area and get their associated lengths
            sorted_boxes = sorted(bounding_boxes, key=lambda x: calculate_bbox_area(x[0]))
            min_bbox, min_length = sorted_boxes[0]
            mid_bbox, mid_length = sorted_boxes[1]
            max_bbox, max_length = sorted_boxes[2]

        
    
            
            prawn_max_normalized_bbox = [max_bbox[0] / 5312, max_bbox[1] / 2988, max_bbox[2] / 5312, max_bbox[3] / 2988]

            prawn_min_normalized_bbox = [min_bbox[0] / 5312, min_bbox[1] / 2988, min_bbox[2] / 5312, min_bbox[3] / 2988]

            prawn_mid_normalized_bbox = [mid_bbox[0] / 5312, mid_bbox[1] / 2988, mid_bbox[2] / 5312, mid_bbox[3] / 2988]
            # true_detections.append(fo.Detection(label="prawn_true", bounding_box=prawn_normalized_bbox))

            closest_detection,ground = find_closest_detection(sample, min_bbox)

            if closest_detection is not None:
                process_detection(closest_detection, sample, filename, prawn_id, filtered_df,ground)

            x_min_max=prawn_max_normalized_bbox[0]
            y_min_max=prawn_max_normalized_bbox[1]
            width_max=prawn_max_normalized_bbox[2]
            heigh_maxt=prawn_max_normalized_bbox[3]

            # Corners in normalized coordinates
            top_left_max = [x_min_max, y_min_max]
            top_right_max = [x_min_max + width_max, y_min_max]
            bottom_left_max = [x_min_max, y_min_max + heigh_maxt]
            bottom_right_max = [x_min_max + width_max, y_min_max + heigh_maxt]

            # Diagonals
            diagonal1_max = [top_left_max, bottom_right_max]
            diagonal2_max = [top_right_max, bottom_left_max]
            

            # diagonal1_polyline_max = fo.Polyline(
            #     label="Diagonal 1",
            #     points=[diagonal1_max],
            #     closed=False,
            #     filled=False,
            #     line_color="red",
            #     thickness=2,
            # )

            # diagonal2_polyline_max = fo.Polyline(
            #     label="longest diagonal",
            #     points=[diagonal2_max],
            #     closed=False,
            #     filled=False,
            #     line_color="yellow",
            #     thickness=2,
            # )

            # max_diagonal_line_1.append(diagonal1_polyline_max)
            # max_diagonal_line_2.append(diagonal2_polyline_max)

            # sample["max_diagonal_line_1"] = fo.Polylines(polylines=max_diagonal_line_1)
            # sample["max_diagonal_line_2"] = fo.Polylines(polylines=max_diagonal_line_2)

        


        ##add tooltip of length_1, length_2, length_3 to the diagonals according to bbox key


            x_min_mid=prawn_mid_normalized_bbox[0]
            y_min_mid=prawn_mid_normalized_bbox[1]
            width_mid=prawn_mid_normalized_bbox[2]
            height_mid=prawn_mid_normalized_bbox[3]

            top_left_mid = [x_min_mid, y_min_mid]
            top_right_mid = [x_min_mid + width_mid, y_min_mid]
            bottom_left_mid = [x_min_mid, y_min_mid + height_mid]
            bottom_right_mid = [x_min_mid + width_mid, y_min_mid + height_mid]

            diagonal1_mid = [top_left_mid, bottom_right_mid]
            diagonal2_mid = [top_right_mid, bottom_left_mid]

            # diagonal1_polyline_mid = fo.Polyline(
            #     label="Diagonal 1",
            #     points=[diagonal1_mid],
            #     closed=False,
            #     filled=False,
            #     line_color="blue",
            #     thickness=2,
            # )

            # diagonal2_polyline_mid = fo.Polyline(
            #     label="shortest diagonal",
            #     points=[diagonal2_mid],
            #     closed=False,
            #     filled=False,
            #     line_color="green",
            #     thickness=2,
            # )

            # mid_diagonal_line_1.append(diagonal1_polyline_mid)
            # mid_diagonal_line_2.append(diagonal2_polyline_mid)

            # sample["mid_diagonal_line_1"] = fo.Polylines(polylines=mid_diagonal_line_1)
            # sample["mid_diagonal_line_2"] = fo.Polylines(polylines=mid_diagonal_line_2)


            # Normalize the largest bounding box coordinates

            x_min = prawn_min_normalized_bbox[0]
            y_min = prawn_min_normalized_bbox[1]
            width = prawn_min_normalized_bbox[2]
            height = prawn_min_normalized_bbox[3]

            # Corners in normalized coordinates
            top_left = [x_min, y_min]
            top_right = [x_min + width, y_min]
            bottom_left = [x_min, y_min + height]
            bottom_right = [x_min + width, y_min + height]

            # Diagonals
            min_diagonal1 = [top_left, bottom_right]
            min_diagonal2 = [top_right, bottom_left]

            # #take the longest diagonal
            # if calculate_euclidean_distance(top_left, bottom_right) > calculate_euclidean_distance(top_right, bottom_left):
            #     longest_diagonal = diagonal1
            # else:
            #     longest_diagonal = diagonal2


            # diagonal1_polyline = fo.Polyline(
            #     label="Diagonal 1",
            #     points=[min_diagonal1],
            #     closed=False,
            #     filled=False,
            #     line_color="blue",
            #     thickness=2,
            # )

            # diagonal2_polyline = fo.Polyline(
            #     label="longest diagonal",
            #     points=[min_diagonal2],
            #     closed=False,
            #     filled=False,
            #     line_color="green",
            #     thickness=2,
            # )


            # min_diagonal_line_1.append(diagonal1_polyline)


            # min_diagonal_line_2.append(diagonal2_polyline)


            # sample["min_diagonal_line_1"] = fo.Polylines(polylines=min_diagonal_line_1)
            # sample["min_diagonal_line_2"] = fo.Polylines(polylines=min_diagonal_line_2)

            diagonal1_polyline_max = fo.Polyline(
                label=f"Max diagonal 1 - Length: {max_length:.2f}mm",
                points=[diagonal1_max],
                closed=False,
                filled=False,
                line_color="red",
                thickness=2,
            )

            diagonal1_polyline_max.label= f'{max_length:.2f}mm'

            diagonal2_polyline_max = fo.Polyline(
                label=f"Max diagonal 2 - Length: {max_length:.2f}mm",
                points=[diagonal2_max],
                closed=False,
                filled=False,
                line_color="yellow",
                thickness=2,
            )
            diagonal2_polyline_max.label= f'{max_length:.2f}mm'

            # For mid bounding box diagonals
            diagonal1_polyline_mid = fo.Polyline(
                label=f"Mid diagonal 1 - Length: {mid_length:.2f}mm",
                points=[diagonal1_mid],
                closed=False,
                filled=False,
                line_color="blue",
                thickness=2,
            )

            diagonal1_polyline_mid.label= f'{mid_length:.2f}mm'

            diagonal2_polyline_mid = fo.Polyline(
                label=f"Mid diagonal 2 - Length: {mid_length:.2f}mm",
                points=[diagonal2_mid],
                closed=False,
                filled=False,
                line_color="green",
                thickness=2,
            )

            diagonal2_polyline_mid.label=f'{mid_length:.2f}mm'

            # For min bounding box diagonals
            diagonal1_polyline = fo.Polyline(
                label=f"Min diagonal 1 - Length: {min_length:.2f}mm",
                points=[min_diagonal1],
                closed=False,
                filled=False,
                line_color="blue",
                thickness=2,
            )

            diagonal1_polyline.label= f'{min_length:.2f}mm' 

            diagonal2_polyline = fo.Polyline(
                label=f"Min diagonal 2 - Length: {min_length:.2f}mm",
                points=[min_diagonal2],
                closed=False,
                filled=False,
                line_color="green",
                thickness=2,
        )
            
            diagonal2_polyline.label= f'{min_length:.2f}mm' 

            min_diagonal_line_1.append(diagonal1_polyline)
            min_diagonal_line_2.append(diagonal2_polyline)  

            max_diagonal_line_1.append(diagonal1_polyline_max)
            max_diagonal_line_2.append(diagonal2_polyline_max)

            mid_diagonal_line_1.append(diagonal1_polyline_mid)
            mid_diagonal_line_2.append(diagonal2_polyline_mid)


            sample["min_diagonal_line_1"] = fo.Polylines(polylines=min_diagonal_line_1)
            sample["min_diagonal_line_2"] = fo.Polylines(polylines=min_diagonal_line_2)
            sample["max_diagonal_line_1"] = fo.Polylines(polylines=max_diagonal_line_1)
            sample["max_diagonal_line_2"] = fo.Polylines(polylines=max_diagonal_line_2)
            sample["mid_diagonal_line_1"] = fo.Polylines(polylines=mid_diagonal_line_1)
            sample["mid_diagonal_line_2"] = fo.Polylines(polylines=mid_diagonal_line_2)


    #add length tooltip to the diagonals according to length_1, length_2, length_3 
   


    # sample["true_detections"] = fo.Detections(detections=true_detections)

def find_closest_detection(sample, prawn_bbox):
    """
    Find closest YOLO detection to a ground truth bounding box.

    Args:
        sample (fo.Sample): FiftyOne sample containing detections
        prawn_bbox (tuple): Ground truth bounding box coordinates (x, y, w, h)

    Returns:
        tuple: (closest_detection_pred, closest_detection_ground_truth)
            - closest_detection_pred: Closest predicted detection
            - closest_detection_ground_truth: Corresponding ground truth
    """
    prawn_point = (prawn_bbox[0] / 5312, prawn_bbox[1] / 2988)
    min_distance = float('inf')
    closest_detection_pred = None
    closest_detection_ground_truth = None  # Initialize to None
    
    # Loop through predicted detections
    for detection_bbox in sample["detections_predictions"].detections:
        det_point = (detection_bbox.bounding_box[0], detection_bbox.bounding_box[1])
        distance = calculate_euclidean_distance(prawn_point, det_point)
        if distance < min_distance:
            min_distance = distance
            closest_detection_pred = detection_bbox
    min_distance = float('inf')
    # Loop through ground truth detections
    for detection_bbox_ground_truth in sample["ground_truth"].detections:
        det_point = (detection_bbox_ground_truth.bounding_box[0], detection_bbox_ground_truth.bounding_box[1])
        distance = calculate_euclidean_distance(prawn_point, det_point)
        if distance < min_distance:
            min_distance = distance
            closest_detection_ground_truth = detection_bbox_ground_truth
    
    # Ensure both closest detections are returned
    return closest_detection_pred, closest_detection_ground_truth


def process_detection(closest_detection, sample, filename, prawn_id, filtered_df,ground ):
    """
    Process matched detections and calculate measurements.

    Args:
        closest_detection (fo.Detection): Matched YOLO detection
        sample (fo.Sample): FiftyOne sample
        filename (str): Image filename
        prawn_id (str): Unique prawn identifier
        filtered_df (pd.DataFrame): DataFrame for storing results
        ground (fo.Detection): Ground truth detection

    Updates:
        - filtered_df: Adds calculated measurements and errors
        - sample: Adds visualization tags based on error thresholds
    """
    height_mm = sample['height(mm)']  # Fixed typo
    if sample.tags[0] == 'test-left' or sample.tags[0] == 'test-right':
        focal_length = 23.64
    else:
        focal_length = 24.72


    # focal_length = 24.22  # Camera focal length
    pixel_size = 0.00716844  # Pixel size in mm

    





    keypoints_dict = closest_detection.attributes["keypoints"]
    carapace_points = [keypoints_dict['start_carapace'], keypoints_dict['eyes']]
    total_length_points = [keypoints_dict['tail'], keypoints_dict['rostrum']]

    keypoint_id=keypoints_dict['keypoint_ID']   

    keypoint1_scaled = [carapace_points[0][0] * 5312, carapace_points[0][1] * 2988]
    keypoint2_scaled = [carapace_points[1][0] * 5312, carapace_points[1][1] * 2988]

    euclidean_distance_pixels = calculate_euclidean_distance(keypoint1_scaled, keypoint2_scaled)
    focal_real_length_cm = calculate_real_width(focal_length, height_mm, euclidean_distance_pixels, pixel_size)
    
    
    object_length_measurer = ObjectLengthMeasurer(5312, 2988, 75.2, 46, height_mm)
    
    distance_mm, angle_deg, distance_px = object_length_measurer.compute_length_two_points(keypoint1_scaled, keypoint2_scaled)
    


    #distance in pixels using df['BoundingBox_1'](B_x 2996.016573,B_y 737.0050179, B_w 159.9996606, B_h 166.0006386)
    #get the top left and bottom right of the bounding box

    bbox = ast.literal_eval(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'BoundingBox_1'].values[0])
    bbox = tuple(float(coord) for coord in bbox)
            
    x_min = bbox[0]
    y_min = bbox[1]
    width = bbox[2]
    height = bbox[3]



    top_left_1 = [x_min, y_min]
    top_right_1 = [x_min + width, y_min]
    bottom_left_1 = [x_min, y_min + height]
    bottom_right_1 = [x_min + width, y_min + height]


    bbox_2 = ast.literal_eval(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'BoundingBox_2'].values[0])
    bbox_2 = tuple(float(coord) for coord in bbox_2)

    x_min_2 = bbox_2[0]
    y_min_2 = bbox_2[1]
    width_2 = bbox_2[2]
    height_2 = bbox_2[3]

    top_left_2 = [x_min_2, y_min_2]
    top_right_2 = [x_min_2 + width_2, y_min_2]
    bottom_left_2 = [x_min_2, y_min_2 + height_2]
    bottom_right_2 = [x_min_2 + width_2, y_min_2 + height_2]

    bbox_3 = ast.literal_eval(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'BoundingBox_3'].values[0])
    bbox_3 = tuple(float(coord) for coord in bbox_3)

    x_min_3 = bbox_3[0]
    y_min_3 = bbox_3[1]
    width_3 = bbox_3[2]
    height_3 = bbox_3[3]

    top_left_3 = [x_min_3, y_min_3]
    top_right_3 = [x_min_3 + width_3, y_min_3]
    bottom_left_3 = [x_min_3, y_min_3 + height_3]
    bottom_right_3 = [x_min_3 + width_3, y_min_3 + height_3]

    distance_px_bounding_box_3 = calculate_euclidean_distance(top_left_3, bottom_right_3)
    distance_px_bounding_box_3_top_right_bottom_left = calculate_euclidean_distance(top_right_3, bottom_left_3)

    


    
    
    
    
    
    


    distance_px_bounding_box_2 = calculate_euclidean_distance(top_left_2, bottom_right_2)
    #top right to bottom left
    distance_px_bounding_box_2_top_right_bottom_left = calculate_euclidean_distance(top_right_2, bottom_left_2)



    distance_px_bounding_box_1 = calculate_euclidean_distance(top_left_1, bottom_right_1)
    #top right to bottom left
    distance_px_bounding_box_1_top_right_bottom_left = calculate_euclidean_distance(top_right_1, bottom_left_1)



    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_bounding_box_1_1'] = distance_px_bounding_box_1_top_right_bottom_left
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_bounding_box_1_2'] = distance_px_bounding_box_1


    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_bounding_box_2_1'] = distance_px_bounding_box_2_top_right_bottom_left
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_bounding_box_2_2'] = distance_px_bounding_box_2



    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_bounding_box_3_1'] = distance_px_bounding_box_3_top_right_bottom_left
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_bounding_box_3_2'] = distance_px_bounding_box_3

    




    ####
    keypoints_dict_ground = ground.attributes["keypoints"]
    keypoints_ground = [keypoints_dict_ground['start_carapace'], keypoints_dict_ground['eyes']]
    keypoint1_scaled_ground = [keypoints_ground[0][0] * 5312, keypoints_ground[0][1] * 2988]
    keypoint2_scaled_ground = [keypoints_ground[1][0] * 5312, keypoints_ground[1][1] * 2988]

    object_length_measurer_ground = ObjectLengthMeasurer(5312, 2988, 75.2, 46, height_mm)

    distance_mm_ground, angle_deg_ground, distance_px_ground = object_length_measurer_ground.compute_length_two_points(keypoint1_scaled_ground, keypoint2_scaled_ground)
    
    #distance_mm_ground to table
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_ground_truth_annotation(mm)'] = distance_mm_ground
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_ground_truth_annotation_pixels'] = distance_px_ground
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'pred_Distance_pixels'] = distance_px

    
    
    # fov=75.2
    # FOV_width=2*height_mm*math.tan(math.radians(fov/2))
    # length_fov=FOV_width*euclidean_distance_pixels/5312

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'id'] = keypoint_id

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_fov(mm)'] = distance_mm

    #add height to the dataframe
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Height(mm)'] = height_mm



    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'focal_RealLength(cm)'] = focal_real_length_cm

    
    min_true_length= min(abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))

    max_true_length= max(abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))

    #take the median of the three lengths
    median_true_length= (abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0])+abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0])+abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0])-min_true_length-max_true_length)


    #save each length_1, length_2, length_3 in pixels to dataframe , Lenght_1_pixels=(Length_1*scale_1)/10

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1_pixels'] = (abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0])*abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Scale_1'].values[0]))/10
                                                                                                                                   
    #lenght_2 in pixels

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2_pixels'] = (abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0])*abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Scale_2'].values[0]))/10

    #lenght_3 in pixels
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3_pixels'] = (abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0])*abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Scale_3'].values[0]))/10
                                                                                                                                                                                                                                                                    



    # true_length = filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Avg_Length'].values[0]

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Pond_Type'] = sample.tags[0]        

    #add euclidean distance in pixels
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Euclidean_Distance'] = euclidean_distance_pixels

    error_percentage_min = abs(distance_mm - min_true_length) / min_true_length * 100
    error_percentage_max = abs(distance_mm - max_true_length) / max_true_length * 100
    error_percentage_median = abs(distance_mm - median_true_length) / median_true_length * 100


    min_error_percentage = min(error_percentage_min, error_percentage_max, error_percentage_median)

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_min'] = error_percentage_min
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_max'] = error_percentage_max
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_median'] = error_percentage_median
    

    #abs error fov
    abs_error_min_fov = abs(distance_mm - min_true_length)
    abs_error_max_fov = abs(distance_mm - max_true_length)
    abs_error_median_fov = abs(distance_mm - median_true_length)

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_fov_min'] = abs_error_min_fov
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_fov_max'] = abs_error_max_fov
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_fov_median'] = abs_error_median_fov

    #abs error focal
    abs_error_min_focal = abs(focal_real_length_cm - min_true_length)
    abs_error_max_focal = abs(focal_real_length_cm - max_true_length)
    abs_error_median_focal = abs(focal_real_length_cm - median_true_length)

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_focal_min'] = abs_error_min_focal
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_focal_max'] = abs_error_max_focal
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_focal_median'] = abs_error_median_focal



    error_focal_min_precentage = abs(focal_real_length_cm - min_true_length) / min_true_length * 100
    error_focal_max_precentage = abs(focal_real_length_cm - max_true_length) / max_true_length * 100
    error_focal_median_precentage = abs(focal_real_length_cm - median_true_length) / median_true_length * 100

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_focal_min'] = error_focal_min_precentage
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_focal_max'] = error_focal_max_precentage
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_focal_median'] = error_focal_median_precentage


    #error distance_mm_ground length_1_length_2_length_3
    error_distance_mm_ground_min = abs(distance_mm_ground - min_true_length)
    error_distance_mm_ground_max = abs(distance_mm_ground - max_true_length)
    error_distance_mm_ground_median = abs(distance_mm_ground - median_true_length)

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_distance_mm_ground_min'] = error_distance_mm_ground_min
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_distance_mm_ground_max'] = error_distance_mm_ground_max
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_distance_mm_ground_median'] = error_distance_mm_ground_median

    #error percentage distance_mm_ground length_1_length_2_length_3
    error_percentage_distance_mm_ground_min = abs(distance_mm_ground - min_true_length) / min_true_length * 100
    error_percentage_distance_mm_ground_max = abs(distance_mm_ground - max_true_length) / max_true_length * 100
    error_percentage_distance_mm_ground_median = abs(distance_mm_ground - median_true_length) / median_true_length * 100

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_distance_mm_ground_min'] = error_percentage_distance_mm_ground_min
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_distance_mm_ground_max'] = error_percentage_distance_mm_ground_max
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_distance_mm_ground_median'] = error_percentage_distance_mm_ground_median




    #ground truth detection label
    ground_truth_detection_label = f'prawn_truth{distance_mm_ground:.2f}mm'

    ground.label = ground_truth_detection_label    


    closest_detection_label = f'pred_length: {distance_mm:.2f}mm'
    
    
    closest_detection.label = closest_detection_label
    closest_detection.attributes["prawn_id"] =fo.Attribute(value=prawn_id)
    # if abs(focal_real_length_cm - min_true_from_length_1_length_2_length_3) / min_true_from_length_1_length_2_length_3 * 100 > 25:
    #     if "MPE_focal>25" not in sample.tags:
    #         sample.tags.append("MPE_focal>25")



    if min_error_percentage> 50:
        if "MPE_fov>50" not in sample.tags:
            sample.tags.append("MPE_fov>50")


    if min_error_percentage > 25 and min_error_percentage <= 50:
        if "MPE_fov>25" not in sample.tags:
            sample.tags.append("MPE_fov>25")

    if min_error_percentage > 10 and min_error_percentage <= 25:
        if "MPE_fov>10" not in sample.tags:
            sample.tags.append("MPE_fov>10")

    if min_error_percentage > 5 and min_error_percentage <= 10:
        if "MPE_fov>5" not in sample.tags:
            sample.tags.append("MPE_fov>5")

    if min_error_percentage <= 5:
        if "MPE_fov<5" not in sample.tags:
            sample.tags.append("MPE_fov<5")
    

# No close match found

def process_images(image_paths, prediction_folder_path, ground_truth_paths_text, filtered_df, metadata_df, dataset,pond_type, measurement_type ):
    """
    Main pipeline for processing images and predictions.

    Args:
        image_paths (list): List of image file paths
        prediction_folder_path (str): Path to YOLO prediction files
        ground_truth_paths_text (list): List of ground truth annotation files
        filtered_df (pd.DataFrame): DataFrame for storing results
        metadata_df (pd.DataFrame): Camera and setup metadata
        dataset (fo.Dataset): FiftyOne dataset
        pond_type (str): Type of pond (test-left, test-right, test-car)

    Saves:
        - Updated filtered_df to Excel
        - Processed samples to FiftyOne dataset
    """
   
    for image_path in tqdm(image_paths):



        
        identifier = extract_identifier(image_path)
        if not identifier:
            print(f"Warning: Could not extract identifier from {image_path}")
            continue
            
        #find the GX010179_200_3927 like pattern like in the filename

            
            # e.g., undistorted_GX010152_36_378.jpg_gamma
        # identifier = filename.replace('undistorted_', '').replace('.jpg_gamma', '')  # Extract the identifier from the filename


        # Construct the paths to the prediction and ground truth files


       #find the filename in the prediction_folder_path using the identifier

       
        # prediction_txt_path = os.path.join(prediction_folder_path, f"{identifier}.txt")


        prediction_txt_path = None  
        for pred_file in os.listdir(prediction_folder_path):
            if identifier in pred_file:
                prediction_txt_path = os.path.join(prediction_folder_path, pred_file)
                break
        if prediction_txt_path is None:
            print(f"No prediction file found for {identifier}")
            continue


        for gt_file in ground_truth_paths_text:
            if identifier in gt_file:
                ground_truth_txt_path = gt_file
                break

        ground_truth_txt_path = None
        for gt_file in ground_truth_paths_text:
            b= extract_identifier((gt_file))
            if b == identifier:
                ground_truth_txt_path = gt_file

                break
        if ground_truth_txt_path is None:
            print(f"No ground truth found for {identifier}")
            continue



        pose_estimations = parse_pose_estimation(prediction_txt_path)


        ground_truths = parse_pose_estimation(ground_truth_txt_path)

        # Process the pose estimations
        keypoints_list, detections = process_poses(pose_estimations)

        keypoints_list_truth, detections_truth = process_poses(ground_truths, is_ground_truth=True)

        sample = fo.Sample(filepath=image_path)
        sample["ground_truth"] = fo.Detections(detections=detections_truth)
        sample["detections_predictions"] = fo.Detections(detections=detections)
        sample["keypoints"] = fo.Keypoints(keypoints=keypoints_list)
        sample["keypoints_truth"] = fo.Keypoints(keypoints=keypoints_list_truth)

        sample.tags.append(pond_type)

        if measurement_type == 'carapace':
            add_metadata(sample, identifier, filtered_df, metadata_df)
        else:
            add_metadata_body(sample, identifier, filtered_df, metadata_df)



        dataset.add_sample(sample)
    if measurement_type == 'carapace':  
        output_file_path = r'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/Updated_carapace_Filtered_Data_with_real_length.xlsx' 
    else:
        output_file_path = r'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/Updated_full_body_Filtered_Data_with_real_length.xlsx' 

        # print(filtered_df.columns) # Change this path accordingly
    filtered_df.to_excel(output_file_path, index=False)

    return filtered_df



def process_images_body(image_paths, prediction_txt_path, ground_truth_paths_text, filtered_df, metadata_df, dataset,pond_type):

    for image_path in tqdm(image_paths):



        filename = os.path.basename(image_path)
        base_filename = os.path.splitext(filename)[0]
        identifier = extract_identifier(filename)
        if not identifier:
            print(f"Warning: Could not extract identifier from {filename}")
            continue
            
            
            # e.g., undistorted_GX010152_36_378.jpg_gamma
        identifier = filename.replace('undistorted_', '').replace('.jpg_gamma', '')  # Extract the identifier from the filename


        # Construct the paths to the prediction and ground truth files
        prediction_file = os.path.join(prediction_txt_path, f"{base_filename}.txt")

        for gt_file in ground_truth_paths_text:
            if filename in gt_file:
                ground_truth_txt_path = gt_file
                break

        ground_truth_txt_path = None
        for gt_file in ground_truth_paths_text:
            b= extract_identifier_from_gt(os.path.basename(gt_file))
            if b == identifier:
                ground_truth_txt_path = gt_file

                break
        if ground_truth_txt_path is None:
            print(f"No ground truth found for {filename}")
            continue

        if not os.path.exists(prediction_file):
            print(f"No prediction file found for {filename}")
            continue
        
        if not os.path.exists(ground_truth_txt_path):
            print(f"No ground truth found for {filename}")
            continue

        ground_truth_txt_path = os.path.join(ground_truth_paths_text, f"{base_filename}.txt")


        pose_estimations = parse_pose_estimation(prediction_file)
        ground_truths = parse_pose_estimation(ground_truth_txt_path)
        
        # Process the pose estimations
        keypoints_list, detections = process_poses(pose_estimations)

        keypoints_list_truth, detections_truth = process_poses(ground_truths, is_ground_truth=True)

        sample = fo.Sample(filepath=image_path)
        sample["ground_truth"] = fo.Detections(detections=detections_truth)
        sample["detections_predictions"] = fo.Detections(detections=detections)
        sample["keypoints"] = fo.Keypoints(keypoints=keypoints_list)
        sample["keypoints_truth"] = fo.Keypoints(keypoints=keypoints_list_truth)

        sample.tags.append(pond_type)


        for file in filtered_df['Label'].unique():
            if identifier in file.split(':')[1]:
                filename=file.split(':')[1]
                break


        print(f'this is the filename {filename}')

        add_metadata_body(sample, filename, filtered_df, metadata_df)



        dataset.add_sample(sample)



        output_file_path = r'Updated_Filtered_Data_with_real_length.xlsx' 

        # print(filtered_df.columns) # Change this path accordingly
        filtered_df.to_excel(output_file_path, index=False)

def process_detection_body(closest_detection, sample, filename, prawn_id, filtered_df,ground ):
    """
    Process matched detections and calculate measurements.

    Args:
        closest_detection (fo.Detection): Matched YOLO detection
        sample (fo.Sample): FiftyOne sample
        filename (str): Image filename
        prawn_id (str): Unique prawn identifier
        filtered_df (pd.DataFrame): DataFrame for storing results
        ground (fo.Detection): Ground truth detection

    Updates:
        - filtered_df: Adds calculated measurements and errors
        - sample: Adds visualization tags based on error thresholds
    """
    
    height_mm = sample['height(mm)']  # Fixed typo
    if sample.tags[0] == 'test-left' or sample.tags[0] == 'test-right':
        focal_length = 23.64
    else:
        focal_length = 24.72


    # focal_length = 24.22  # Camera focal length
    pixel_size = 0.00716844  # Pixel size in mm

    





    keypoints_dict = closest_detection.attributes["keypoints"]
    # carapace_points = [keypoints_dict['start_carapace'], keypoints_dict['eyes']]
    total_length_points = [keypoints_dict['tail'], keypoints_dict['rostrum']]

    keypoint_id=keypoints_dict['keypoint_ID']   

    keypoint1_scaled = [total_length_points[0][0] * 5312, total_length_points[0][1] * 2988]
    keypoint2_scaled = [total_length_points[1][0] * 5312, total_length_points[1][1] * 2988]

    euclidean_distance_pixels = calculate_euclidean_distance(keypoint1_scaled, keypoint2_scaled)
    focal_real_length_cm = calculate_real_width(focal_length, height_mm, euclidean_distance_pixels, pixel_size)
    
    
    object_length_measurer = ObjectLengthMeasurer(5312, 2988, 75.2, 46, height_mm)
    
    distance_mm, angle_deg, distance_px = object_length_measurer.compute_length_two_points(keypoint1_scaled, keypoint2_scaled)
    
    ####
    keypoints_dict_ground = ground.attributes["keypoints"]
    keypoints_ground = [keypoints_dict_ground['tail'], keypoints_dict_ground['rostrum']]
    keypoint1_scaled_ground = [keypoints_ground[0][0] * 5312, keypoints_ground[0][1] * 2988]
    keypoint2_scaled_ground = [keypoints_ground[1][0] * 5312, keypoints_ground[1][1] * 2988]

    object_length_measurer_ground = ObjectLengthMeasurer(5312, 2988, 75.2, 46, height_mm)

    distance_mm_ground, angle_deg_ground, distance_px_ground = object_length_measurer_ground.compute_length_two_points(keypoint1_scaled_ground, keypoint2_scaled_ground)
    
    #distance_mm_ground to table
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_ground_truth_annotation(mm)'] = distance_mm_ground
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_ground_truth_annotation_pixels'] = distance_px_ground
    
    #distnace in pixels to table
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'pred_Distance_pixels'] = distance_px
    
    # fov=75.2
    # FOV_width=2*height_mm*math.tan(math.radians(fov/2))
    # length_fov=FOV_width*euclidean_distance_pixels/5312

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'id'] = keypoint_id

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_fov(mm)'] = distance_mm

    #add height to the dataframe
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Height(mm)'] = height_mm



    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'focal_RealLength(cm)'] = focal_real_length_cm
    
    print(f'{filename}  {prawn_id} ')
    
    min_true_length= min(abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))

    max_true_length= max(abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))

    #take the median of the three lengths
    median_true_length= (abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0])+abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0])+abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0])-min_true_length-max_true_length)


    #save each length_1, length_2, length_3 in pixels to dataframe , Lenght_1_pixels=(Length_1*scale_1)/10

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1_pixels'] = (abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0])*abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Scale_1'].values[0]))/10
                                                                                                                                   
    #lenght_2 in pixels

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2_pixels'] = (abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0])*abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Scale_2'].values[0]))/10

    #lenght_3 in pixels
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3_pixels'] = (abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0])*abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Scale_3'].values[0]))/10
                                                                                                                                                                                                                                                                    



    # true_length = filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Avg_Length'].values[0]

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Pond_Type'] = sample.tags[0]        

    #add euclidean distance in pixels
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Euclidean_Distance'] = euclidean_distance_pixels

    error_percentage_min = abs(distance_mm - min_true_length) / min_true_length * 100
    error_percentage_max = abs(distance_mm - max_true_length) / max_true_length * 100
    error_percentage_median = abs(distance_mm - median_true_length) / median_true_length * 100


    min_error_percentage = min(error_percentage_min, error_percentage_max, error_percentage_median)

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_min'] = error_percentage_min
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_max'] = error_percentage_max
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_fov_median'] = error_percentage_median
    

    #abs error fov
    abs_error_min_fov = abs(distance_mm - min_true_length)
    abs_error_max_fov = abs(distance_mm - max_true_length)
    abs_error_median_fov = abs(distance_mm - median_true_length)

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_fov_min'] = abs_error_min_fov
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_fov_max'] = abs_error_max_fov
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_fov_median'] = abs_error_median_fov

    #abs error focal
    abs_error_min_focal = abs(focal_real_length_cm - min_true_length)
    abs_error_max_focal = abs(focal_real_length_cm - max_true_length)
    abs_error_median_focal = abs(focal_real_length_cm - median_true_length)

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_focal_min'] = abs_error_min_focal
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_focal_max'] = abs_error_max_focal
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'AbsError_focal_median'] = abs_error_median_focal



    error_focal_min_precentage = abs(focal_real_length_cm - min_true_length) / min_true_length * 100
    error_focal_max_precentage = abs(focal_real_length_cm - max_true_length) / max_true_length * 100
    error_focal_median_precentage = abs(focal_real_length_cm - median_true_length) / median_true_length * 100

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_focal_min'] = error_focal_min_precentage
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_focal_max'] = error_focal_max_precentage
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'MPError_focal_median'] = error_focal_median_precentage


    #error distance_mm_ground length_1_length_2_length_3
    error_distance_mm_ground_min = abs(distance_mm_ground - min_true_length)
    error_distance_mm_ground_max = abs(distance_mm_ground - max_true_length)
    error_distance_mm_ground_median = abs(distance_mm_ground - median_true_length)

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_distance_mm_ground_min'] = error_distance_mm_ground_min
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_distance_mm_ground_max'] = error_distance_mm_ground_max
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_distance_mm_ground_median'] = error_distance_mm_ground_median

    #error percentage distance_mm_ground length_1_length_2_length_3
    error_percentage_distance_mm_ground_min = abs(distance_mm_ground - min_true_length) / min_true_length * 100
    error_percentage_distance_mm_ground_max = abs(distance_mm_ground - max_true_length) / max_true_length * 100
    error_percentage_distance_mm_ground_median = abs(distance_mm_ground - median_true_length) / median_true_length * 100

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_distance_mm_ground_min'] = error_percentage_distance_mm_ground_min
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_distance_mm_ground_max'] = error_percentage_distance_mm_ground_max
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_distance_mm_ground_median'] = error_percentage_distance_mm_ground_median



    #ground truth detection label
    ground_truth_detection_label = f'prawn_truth{distance_mm_ground:.2f}mm'

    ground.label = ground_truth_detection_label    



    closest_detection_label = f'pred_length: {distance_mm:.2f}mm' 
    
    
    
    closest_detection.label = closest_detection_label
    closest_detection.attributes["prawn_id"] =fo.Attribute(value=prawn_id)
    # if abs(focal_real_length_cm - min_true_from_length_1_length_2_length_3) / min_true_from_length_1_length_2_length_3 * 100 > 25:
    #     if "MPE_focal>25" not in sample.tags:
    #         sample.tags.append("MPE_focal>25")



    if min_error_percentage> 50:
        if "MPE_fov>50" not in sample.tags:
            sample.tags.append("MPE_fov>50")


    if min_error_percentage > 25 and min_error_percentage <= 50:
        if "MPE_fov>25" not in sample.tags:
            sample.tags.append("MPE_fov>25")

    if min_error_percentage > 10 and min_error_percentage <= 25:
        if "MPE_fov>10" not in sample.tags:
            sample.tags.append("MPE_fov>10")

    if min_error_percentage > 5 and min_error_percentage <= 10:
        if "MPE_fov>5" not in sample.tags:
            sample.tags.append("MPE_fov>5")

    if min_error_percentage <= 5:
        if "MPE_fov<5" not in sample.tags:
            sample.tags.append("MPE_fov<5")
    

def extract_identifier(filename):
    """
    Extract identifier pattern like 'GX010179_200_3927' from filename.
    
    Args:
        filename (str): The filename to process
        
    Returns:
        str: The extracted identifier or None if not found
    """
    # Pattern matches: GX followed by digits, underscore, digits, underscore, digits
    pattern = r'(GX\d+_\d+_\d+)'
    match = re.search(pattern, filename)
    return match.group(1) if match else None
