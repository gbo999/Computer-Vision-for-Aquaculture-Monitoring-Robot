# data_loader.py
import fiftyone as fo
import pandas as pd
import os
import ast
from tqdm import tqdm
from utils import parse_pose_estimation, calculate_euclidean_distance, calculate_real_width, extract_identifier_from_gt, calculate_bbox_area
import math


def load_data(filtered_data_path, metadata_path):
    filtered_df = pd.read_csv(filtered_data_path)
    metadata_df = pd.read_excel(metadata_path)
    return filtered_df, metadata_df

def create_dataset():
    dataset = fo.Dataset("prawn_combined_dataset22", overwrite=True)
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
                keypoints_dict = {'point1': keypoints[0], 'point2': keypoints[1]}
                detections.append(fo.Detection(label="prawn", bounding_box=[x1_rel, y1_rel, width_rel, height_rel], attributes={'keypoints': keypoints_dict}))
            else:
                detections.append(fo.Detection(label="prawn_truth", bounding_box=[x1_rel, y1_rel, width_rel, height_rel]))
    
    return keypoints_list, detections

def add_metadata(sample, filename, filtered_df, metadata_df, swimmingdf=None):
    matching_rows = filtered_df[filtered_df['Label'] == f'carapace:{filename}']


    parts = filename.split('_')
    relevant_part = f"{parts[1][-3:]}_{parts[3].split('.')[0]}"
    
    metadata_row = metadata_df[metadata_df['file name'] == relevant_part]

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
    diagonal_line_1=[]
    diagonal_line_2=[]

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
        
        prawn_normalized_bbox = [min_bbox[0] / 5312, min_bbox[1] / 2988, min_bbox[2] / 5312, min_bbox[3] / 2988]

        # true_detections.append(fo.Detection(label="prawn_true", bounding_box=prawn_normalized_bbox))

        closest_detection = find_closest_detection(sample, min_bbox)

        if closest_detection is not None:
            process_detection(closest_detection, sample, filename, prawn_id, filtered_df)

       
        
        # Normalize the largest bounding box coordinates

        x_min = prawn_normalized_bbox[0]
        y_min = prawn_normalized_bbox[1]
        width = prawn_normalized_bbox[2]
        height = prawn_normalized_bbox[3]

        # Corners in normalized coordinates
        top_left = [x_min, y_min]
        top_right = [x_min + width, y_min]
        bottom_left = [x_min, y_min + height]
        bottom_right = [x_min + width, y_min + height]

        # Diagonals
        diagonal1 = [top_left, bottom_right]
        diagonal2 = [top_right, bottom_left]

        # Create polylines for diagonals
        diagonal1_polyline = fo.Polyline(
            label="Diagonal 1",
            points=[diagonal1],
            closed=False,
            filled=False,
            line_color="blue",
            thickness=2,
        )

        diagonal2_polyline = fo.Polyline(
            label="Diagonal 2",
            points=[diagonal2],
            closed=False,
            filled=False,
            line_color="green",
            thickness=2,
        )


        diagonal_line_1.append(diagonal1_polyline)
        diagonal_line_2.append(diagonal2_polyline)


    sample["diagonal_line_1"] = fo.Polylines(polylines=diagonal_line_1)
    sample["diagonal_line_2"] = fo.Polylines(polylines=diagonal_line_2)






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
    if sample.tags[0] == 'pond_1\carapace\left' or sample.tags[0] == 'pond_1\carapace\right':
        focal_length = 23.64
    else:
        focal_length = 24.72


    # focal_length = 24.22  # Camera focal length
    pixel_size = 0.00716844  # Pixel size in mm

    fov=75
    FOV_width=2*height_mm*math.tan(math.radians(fov/2))

    keypoints_dict2 = closest_detection.attributes["keypoints"]
    keypoints1 = [keypoints_dict2['point1'], keypoints_dict2['point2']]

    keypoint1_scaled = [keypoints1[0][0] * 5312, keypoints1[0][1] * 2988]
    keypoint2_scaled = [keypoints1[1][0] * 5312, keypoints1[1][1] * 2988]

    euclidean_distance_pixels = calculate_euclidean_distance(keypoint1_scaled, keypoint2_scaled)
    real_length_cm = calculate_real_width(focal_length, height_mm, euclidean_distance_pixels, pixel_size)

    length_fov=FOV_width*euclidean_distance_pixels/5312

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_fov(mm)'] = length_fov

    #add height to the dataframe
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Height(mm)'] = height_mm



    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'RealLength(cm)'] = real_length_cm

    
    min_true_from_length_1_length_2_length_3= min(abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))
    # true_length = filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Avg_Length'].values[0]

    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Pond_Type'] = sample.tags[0]        

    #add euclidean distance in pixels
    filtered_df.loc[(filtered_df['Label'] == f'carapace:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Euclidean_Distance'] = euclidean_distance_pixels



    closest_detection_label = f'MPError: {abs(real_length_cm - min_true_from_length_1_length_2_length_3) / min_true_from_length_1_length_2_length_3 * 100:.2f}%, true length: {min_true_from_length_1_length_2_length_3:.2f}cm, pred length: {real_length_cm:.2f}cm, mpe_fov: {abs(length_fov - min_true_from_length_1_length_2_length_3) / min_true_from_length_1_length_2_length_3 * 100:.2f}%, pred length: {length_fov:.2f}cm'
    closest_detection.label = closest_detection_label
    closest_detection.attributes["prawn_id"] =fo.Attribute(value=prawn_id)
    if abs(real_length_cm - min_true_from_length_1_length_2_length_3) / min_true_from_length_1_length_2_length_3 * 100 > 25:
        if "MPE_focal>25" not in sample.tags:
            sample.tags.append("MPE_focal>25")

    if abs(length_fov - min_true_from_length_1_length_2_length_3) / min_true_from_length_1_length_2_length_3 * 100 > 50:
        if "MPE_fov>50" not in sample.tags:
            sample.tags.append("MPE_fov>50")


    if abs(length_fov - min_true_from_length_1_length_2_length_3) / min_true_from_length_1_length_2_length_3 * 100 > 25:
        if "MPE_fov>25" not in sample.tags:
            sample.tags.append("MPE_fov>25")
    

# No close match found

def process_images(image_paths, prediction_folder_path, ground_truth_paths_text, filtered_df, metadata_df, dataset,pond_type):
   
   for image_path in tqdm(image_paths):


    filename = os.path.splitext(os.path.basename(image_path))[0] 
     
    
     
     
     # e.g., undistorted_GX010152_36_378.jpg_gamma
    identifier = filename.replace('undistorted_', '').replace('.jpg_gamma', '')  # Extract the identifier from the filename


    # Construct the paths to the prediction and ground truth files
    prediction_txt_path = os.path.join(prediction_folder_path, f"{filename}.txt")

    # Match ground truth based on the extracted identifier
    ground_truth_txt_path = None
    for gt_file in ground_truth_paths_text:
        b= extract_identifier_from_gt(os.path.basename(gt_file))
        if b == identifier:
            ground_truth_txt_path = gt_file

            break
    if ground_truth_txt_path is None:
        print(f"No ground truth found for {filename}")
        continue
    
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
   output_file_path = r'Updated_Filtered_Data_with_real_length.xlsx'  # Change this path accordingly
   filtered_df.to_excel(output_file_path, index=False)
