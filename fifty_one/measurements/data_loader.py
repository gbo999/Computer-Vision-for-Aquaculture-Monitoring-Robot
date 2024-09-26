# data_loader.py
import fiftyone as fo
import pandas as pd
import os
import ast
from tqdm import tqdm
from utils import parse_pose_estimation, calculate_euclidean_distance, calculate_real_width, extract_identifier_from_gt, calculate_bbox_area
import math
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
        if angle < -45:
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
        
        return distance_mm, normalized_angle

def load_data(filtered_data_path, metadata_path):
    filtered_df = pd.read_csv(filtered_data_path)
    metadata_df = pd.read_excel(metadata_path)
    return filtered_df, metadata_df

def create_dataset():
    dataset = fo.Dataset("prawn_combined_dataset444444444444", overwrite=True)
    dataset.default_skeleton = fo.KeypointSkeleton(
        labels=["start_carapace", "eyes"],
        edges=[[0, 1]],
    )
    return dataset

def process_poses(poses, is_ground_truth=False):
    keypoints_list = []
    detections = []

    for pose in poses:
        if len(pose) == 11:
            x1_rel, y1_rel, width_rel, height_rel = pose[1:5]
            x1_rel -= width_rel / 2
            y1_rel -= height_rel / 2

            keypoints = [[pose[i], pose[i + 1]] for i in range(5, len(pose), 3)]
            keypoint = fo.Keypoint(points=keypoints)
            keypoints_list.append(keypoint)

            if not is_ground_truth:
                keypoints_dict = {'point1': keypoints[0], 'point2': keypoints[1],'keypoint_ID':keypoint.id}
                detections.append(fo.Detection(label="prawn", bounding_box=[x1_rel, y1_rel, width_rel, height_rel], attributes={'keypoints': keypoints_dict}))
            else:
                detections.append(fo.Detection(label="prawn_truth", bounding_box=[x1_rel, y1_rel, width_rel, height_rel]))
    
    return keypoints_list, detections

def add_metadata(sample, filename, filtered_df, metadata_df, swimmingdf=None):

    if 'undistorted' in filename:
        filename = filename.replace('undistorted_', '')


    compatible_file_name= filename.split('_')[0:3]

    comp=compatible_file_name[2].split('-')[0]

    compatible_file_name[2]=comp

    print(f'compatible {compatible_file_name}')


    #rows where compatible file nams string in file name
    matching_rows = filtered_df[filtered_df['Label'].str.contains('_'.join(compatible_file_name))]
    
    filename=matching_rows['Label'].values[0].split(':')[1] 

    joined_string ='_'.join([compatible_file_name[0],compatible_file_name[2]])
    


    relevant_part =joined_string 
    
    # relevant_part = str('_'.join(compatible_file_name)[0],str('_'.join(compatible_file_name)[2]))
    
    metadata_row = metadata_df[metadata_df['file_name_new'] == relevant_part]

    if not metadata_row.empty:
        metadata = metadata_row.iloc[0].to_dict() 
        for key, value in metadata.items():
            if key != 'file name':
                sample[key] = value
    else:
        print(f"No metadata found for {relevant_part}")
    
    add_prawn_detections(sample, matching_rows, filtered_df,filename)

def add_prawn_detections(sample, matching_rows, filtered_df,filename):
    # true_detections = []
    min_diagonal_line_1=[]
    min_diagonal_line_2=[]

    max_diagonal_line_1=[]
    max_diagonal_line_2=[]

    for _, row in matching_rows.iterrows():
        prawn_id = row['PrawnID']
        bounding_boxes = []
        for bbox_key in ['BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3']:
            if pd.notna(row[bbox_key]):
                bbox = ast.literal_eval(row[bbox_key])
                bbox = tuple(float(coord) for coord in bbox)  # Convert bounding box to tuple of floats
                bounding_boxes.append(bbox)
        
        if not bounding_boxes:
            print(f"No bounding boxes found for prawn ID {prawn_id} in {filename}.")
            continue

        # Select the largest bounding box based on area
        min_bbox = min(bounding_boxes, key=calculate_bbox_area)
        max_bbox = max(bounding_boxes, key=calculate_bbox_area)    





        prawn_max_normalized_bbox = [max_bbox[0] / 5312, max_bbox[1] / 2988, max_bbox[2] / 5312, max_bbox[3] / 2988]

        prawn_min_normalized_bbox = [min_bbox[0] / 5312, min_bbox[1] / 2988, min_bbox[2] / 5312, min_bbox[3] / 2988]

        # true_detections.append(fo.Detection(label="prawn_true", bounding_box=prawn_normalized_bbox))

        closest_detection = find_closest_detection(sample, min_bbox)

        if closest_detection is not None:
            process_detection(closest_detection, sample, filename, prawn_id, filtered_df)

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
            label="Diagonal 1",
            points=[diagonal1_max],
            closed=False,
            filled=False,
            line_color="red",
            thickness=2,
        )

        diagonal2_polyline_max = fo.Polyline(
            label="longest diagonal",
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


        diagonal1_polyline = fo.Polyline(
            label="Diagonal 1",
            points=[min_diagonal1],
            closed=False,
            filled=False,
            line_color="blue",
            thickness=2,
        )

        diagonal2_polyline = fo.Polyline(
            label="longest diagonal",
            points=[min_diagonal2],
            closed=False,
            filled=False,
            line_color="green",
            thickness=2,
        )


        min_diagonal_line_1.append(diagonal1_polyline)


        min_diagonal_line_2.append(diagonal2_polyline)


    sample["min_diagonal_line_1"] = fo.Polylines(polylines=min_diagonal_line_1)
    sample["min_diagonal_line_2"] = fo.Polylines(polylines=min_diagonal_line_2)






    # sample["true_detections"] = fo.Detections(detections=true_detections)

def find_closest_detection(sample, prawn_bbox):
    prawn_point = (prawn_bbox[0] / 5312, prawn_bbox[1] / 2988)
    min_distance = float('inf')
    closest_detection = None

    for detection_bbox in sample["detections_predictions"].detections:
        det_point = (detection_bbox.bounding_box[0], detection_bbox.bounding_box[1])
        distance = calculate_euclidean_distance(prawn_point, det_point)
        if distance < min_distance:
            min_distance = distance
            closest_detection = detection_bbox
    
    return closest_detection

def process_detection(closest_detection, sample, filename, prawn_id, filtered_df):
    height_mm = sample['heigtht(mm)']
    if sample.tags[0] == 'test-left' or sample.tags[0] == 'test-right':
        focal_length = 23.64
    else:
        focal_length = 24.72


    # focal_length = 24.22  # Camera focal length
    pixel_size = 0.00716844  # Pixel size in mm

    

    keypoints_dict2 = closest_detection.attributes["keypoints"]
    keypoints1 = [keypoints_dict2['point1'], keypoints_dict2['point2']]


    keypoint_id=keypoints_dict2['keypoint_ID']   

    keypoint1_scaled = [keypoints1[0][0] * 5312, keypoints1[0][1] * 2988]
    keypoint2_scaled = [keypoints1[1][0] * 5312, keypoints1[1][1] * 2988]

    euclidean_distance_pixels = calculate_euclidean_distance(keypoint1_scaled, keypoint2_scaled)
    focal_real_length_cm = calculate_real_width(focal_length, height_mm, euclidean_distance_pixels, pixel_size)
    
    
    object_length_measurer = ObjectLengthMeasurer(5312, 2988, 75.2, 46, height_mm)
    
    distance_mm, angle_deg = object_length_measurer.compute_length_two_points(keypoint1_scaled, keypoint2_scaled)
    
    
    
    
    
    
    
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

    closest_detection_label = f'pred_length: {distance_mm},median_length: {median_true_length}  ,MPError_min: {error_percentage_min:.2f}% , MPError_max: {error_percentage_max:.2f}%, MPError_median: {error_percentage_median:.2f}%' 
    
    
    
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

def process_images(image_paths, prediction_folder_path, ground_truth_paths_text, filtered_df, metadata_df, dataset,pond_type):
   
   for image_path in tqdm(image_paths):



    filename = os.path.splitext(os.path.basename(image_path))[0] 
     
     
     
     # e.g., undistorted_GX010152_36_378.jpg_gamma
    # identifier = filename.replace('undistorted_', '').replace('.jpg_gamma', '')  # Extract the identifier from the filename


    # Construct the paths to the prediction and ground truth files
    prediction_txt_path = os.path.join(prediction_folder_path, f"{filename}.txt")

    for gt_file in ground_truth_paths_text:
        if filename in gt_file:
            ground_truth_txt_path = gt_file
            break

    # Match ground truth based on the extracted identifier
    # ground_truth_txt_path = None
    # for gt_file in ground_truth_paths_text:
    #     b= extract_identifier_from_gt(os.path.basename(gt_file))
    #     if b == identifier:
    #         ground_truth_txt_path = gt_file

    #         break
    # if ground_truth_txt_path is None:
    #     print(f"No ground truth found for {filename}")
    #     continue
    
    # Parse the pose estimation data from the TXT file
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
    add_metadata(sample, filename, filtered_df, metadata_df)



    dataset.add_sample(sample)

   output_file_path = r'Updated_Filtered_Data_with_real_length.xlsx' 
   
   print(filtered_df.columns) # Change this path accordingly
   filtered_df.to_excel(output_file_path, index=False)



