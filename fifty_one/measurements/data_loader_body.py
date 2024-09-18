from MEC import welzl
from utils import calculate_euclidean_distance
from utils import calculate_real_width, calculate_bbox_area
import os
import ast
from tqdm import tqdm
import fiftyone as fo
import pandas as pd
from MEC import Point
import fiftyone.core.labels as fol
import cv2
from skeletonization import skeletonize_mask, draw_skeleton, draw_longest_path, find_longest_path,create_filled_binary_mask
import numpy as np


def load_data(filtered_data_path, metadata_path):
    filtered_df = pd.read_excel(filtered_data_path)
    metadata_df = pd.read_excel(metadata_path)
    return filtered_df, metadata_df

def create_dataset():
    dataset = fo.Dataset("prawn_full_body", overwrite=True)
   
    return dataset


def process_segmentations(segmentation_path):
    """
    Process the segmentations from the TXT file, calculate the minimum enclosing circle for each prawn.
    """
    segmentations = []
    skeletons=[]
    hulls=[]
    # Open the segmentation file and process each line
    with open(segmentation_path, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split()))
            binary_mask = create_filled_binary_mask(coords, 640, 640)
            skeleton = skeletonize_mask(binary_mask)
            skeleton_coords = np.column_stack(np.nonzero(skeleton))
            normalized_coords,max_length = find_longest_path(skeleton_coords,(640,640),(2988,5312))

            normalized_coords = [(x, y) for y, x in normalized_coords]  # Convert to (y, x) format

            #convex hull diameter
            contures, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            prawn_conture = max(contures, key=cv2.contourArea)  


            hull_points = cv2.convexHull(prawn_conture, returnPoints=True)


            
# Scaling factors to convert from 640x640 to 5312x2988
            scale_x = 5312 / 640
            scale_y = 2988 / 640

        # Scale the points to the new resolution
            scaled_hull_points = []
            for point in hull_points:
                x, y = point[0]
                scaled_x = x * scale_x
                scaled_y = y * scale_y
                scaled_hull_points.append([scaled_x, scaled_y])

            # Convert to numpy array for easier handling
            scaled_hull_points = np.array(scaled_hull_points, dtype=np.float32)

            # Now, find the maximum Euclidean distance (convex hull diameter) using the scaled points
            max_distance = 0
            point1 = None
            point2 = None

            # Loop through all pairs of scaled points to find the maximum distance
            for i in range(len(scaled_hull_points)):
                for j in range(i + 1, len(scaled_hull_points)):
                    distance = calculate_euclidean_distance(scaled_hull_points[i], scaled_hull_points[j])
                    if distance > max_distance:
                        max_distance = distance
                        point1 = scaled_hull_points[i]
                        point2 = scaled_hull_points[j]

            # The result is max_distance (in pixels) in the 5312x2988 image


            normalzied_points_hull = [(point1[0]/5312, point1[1]/2988), (point2[0]/5312, point2[1]/2988)]  # Extract points (x, y)

            hull=fo.Polyline(
                points=[normalzied_points_hull],
                closed=False,
                filled=False,
                max_length=max_distance
            )

             


            hulls.append(hull)

            skeleton=fo.Polyline(
                points=[normalized_coords],
                closed=False,
                filled=False,
                max_length=max_length
            )

            skeletons.append(skeleton)

              # Convert the line to a list of floats
            normalzied_points = [(coords[i]/640, coords[i + 1]/640) for i in range(0, len(coords), 2)]  # Extract points (x, y)
            points = [Point(x*5312, y*2988) for x, y in normalzied_points] 
            

            
             # Convert to Point objects    
            # Calculate the minimum enclosing circle (center and radius)
            center, radius = calculate_minimum_enclosing_circle(points)
            diameter = radius * 2

            segmentation = fo.Polyline(
                points=[normalzied_points],
                closed=True,
                filled=False,
                diameter=diameter,
                center=center,
                max_length_skeleton=max_length,
                max_length_hull=max_distance
            )




            segmentations.append(segmentation)
                     # Store the segmentation information (center, radius, and diameter)

    return segmentations,skeletons,hulls

def calculate_minimum_enclosing_circle(points):
    """
    Calculate the minimum enclosing circle for a set of points using Welzl's algorithm.
    Returns the center and radius of the circle.
    """
    mec = welzl(points)  
    center=[]
    center.append(mec.C.X)
    center.append(mec.C.Y)
    return center, mec.R
def find_closest_circle_center(prawn_bbox, segmentations):
    """
    Find the closest segmentation circle center to the bounding box (bx, by) using Euclidean distance.
    """
    prawn_point = (prawn_bbox[0], prawn_bbox[1])  # Top-left corner of bounding box (bx, by)
    min_distance = float('inf')
    closest_center = None

    seg=None    
    # Iterate over each segmentation and calculate the distance to the bounding box corner
    for segmentation in segmentations:

        center = segmentation['center']  # Get the circle center (cx, cy)
        distance = calculate_euclidean_distance(prawn_point, center)

        if distance < min_distance:
            min_distance = distance
            closest_center = center
            seg=segmentation
    return seg, min_distance
def process_detection_by_circle(segmentation, sample, filename, prawn_id, filtered_df):
    """
    Process the prawn detection based on the enclosing circle's diameter.
    Update the filtered dataframe with the real-world size of the prawn.
    """

    

    # Fetch height in mm and other metadata
    height_mm = sample['heigtht(mm)'] 
    focal_length = 24.22  # Camera focal length
    pixel_size = 0.00716844  # Pixel size in mm

    poly=segmentation[0]

    # Get the diameter of the circle in pixels
    predicted_diameter_pixels = poly['diameter']


    predicted_skeleton_length=poly['max_length_skeleton']  

    predicted_hull_length=poly['max_length_hull']

     


    hull_length_cm = calculate_real_width(focal_length, height_mm, predicted_hull_length, pixel_size)    

    # Calculate the real-world prawn size using the enclosing circle's diameter
    real_length_cm = calculate_real_width(focal_length, height_mm, predicted_diameter_pixels, pixel_size)

    ske_length_cm = calculate_real_width(focal_length, height_mm, predicted_skeleton_length, pixel_size)    


    #add to filtered dataframe the number of pixels in the hull and skeleton and diameter
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Hull_Length_pixels'] = predicted_hull_length
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Skeleton_Length_pixels'] = predicted_skeleton_length
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Diameter_pixels'] = predicted_diameter_pixels



    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'RealLength_Hull(cm)'] = hull_length_cm

    # Update the filtered dataframe with the calculated real length
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'RealLength_MEC(cm)'] = real_length_cm
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'RealLength_Skeleton(cm)'] = ske_length_cm

    #add pond type to the filtered dataframe
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'PondType'] = sample.tags[0]        

    #put height in mm in the filtered dataframe
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Height_mm'] = height_mm

    # Fetch the true length from the dataframe
    # true_length = filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Avg_Length'].values[0]
    true_length = max(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0],filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0],filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0])
    # Calculate the larges  length from Length_1, Length_2, Length_3

    # true_length = max(lengths)
    # Compute error and create label for the closest detection


    error_percentage = abs(real_length_cm - true_length) / true_length * 100

    error_percentage_skeleton = abs(ske_length_cm - true_length) / true_length * 100    

    error_percentage_hull = abs(hull_length_cm - true_length) / true_length * 100

    closest_detection_label = f'MPError: {error_percentage:.2f}%, true length: {true_length:.2f}cm, pred length: {real_length_cm:.2f}cm ,error percentage skeleton: {error_percentage_skeleton:.2f}%, true length: {true_length:.2f}cm, pred length: {ske_length_cm:.2f}cm, error percentage hull: {error_percentage_hull:.2f}%, true length: {true_length:.2f}cm, pred length: {hull_length_cm:.2f}cm' 
    poly.label = closest_detection_label
    poly.attributes["prawn_id"] = fo.Attribute(value=prawn_id)
    # Attach information to the sample

    # Tagging the sample based on error percentage
    if error_percentage > 25:
        if "MPE_mec>25" not in sample.tags:
            sample.tags.append("MPE_mec>25")

    if error_percentage_skeleton > 25:
        if "MPE_ske>25" not in sample.tags:
            sample.tags.append("MPE_ske>25")
   
def process_images(image_paths, prediction_folder_path, filtered_df, metadata_df, dataset, pond_tag):
    """
    Processes images by matching segmentation with bounding boxes and calculating prawn sizes.
    """
    for image_path in tqdm(image_paths):
        filename = os.path.splitext(os.path.basename(image_path))[0]


        core_name=filename.split('.')[0]
        # Construct the path to the prediction (segmentation) file
        prediction_txt_path = os.path.join(prediction_folder_path, f"{core_name}_segmentations.txt")
        if not os.path.exists(prediction_txt_path):
            print(f"No segmentation file found for {filename}")
            continue


                 
        # Parse the segmentations to get the minimum enclosing circles
        segmentations,skeletons,hulls = process_segmentations(prediction_txt_path)

        # Save the modified image (with circles drawn)
       

        # Load bounding boxes from the filtered data
        matching_rows = filtered_df[filtered_df['Label'] == f'full body:{filename}']
        
        if matching_rows.empty:
            continue


        # Create a new sample for FiftyOne
        sample = fo.Sample(filepath=image_path)

        # Iterate over each bounding box in the filtered data
        sample["segmentations"] = fol.Polylines(polylines=segmentations)

        sample["skeletons"] = fol.Polylines(polylines=skeletons)

        sample["hulls"] = fol.Polylines(polylines=hulls)    

        sample.tags.append(f"pond_{pond_tag}")



        # Add the processed sample to the FiftyOne dataset
        add_metadata(sample, filename, filtered_df, metadata_df)
        dataset.add_sample(sample)

    # Save the updated dataframe with the calculated real lengths
    output_file_path = r'Updated_full_Filtered_Data_with_real_length.xlsx'
    filtered_df.to_excel(output_file_path, index=False)
    print("Processed and saved the updated filtered data.")


def add_metadata(sample, filename, filtered_df, metadata_df, swimmingdf=None):
    """
    Add metadata to a sample in FiftyOne.
    """
    # Filter matching rows from the filtered data
    matching_rows = filtered_df[filtered_df['Label'] == f'full body:{filename}']

    # Extract relevant parts from the filename
    parts = filename.split('_')
    relevant_part = f"{parts[1][-3:]}_{parts[3].split('.')[0]}"
    
    metadata_df['file name'] = metadata_df['file name'].str.strip()

    # Look for metadata based on the relevant part of the filename
    metadata_row = metadata_df[metadata_df['file name'] == relevant_part]

    # Add the metadata to the sample
    if not metadata_row.empty:
        metadata = metadata_row.iloc[0].to_dict()
        for key, value in metadata.items():
            if key != 'file name':  # Skip the 'file name' column
                sample[key] = value
    else:
        print(f"No metadata found for {relevant_part}")
    
    # Add prawn detections using the segmentation data
    add_prawn_detections(sample, matching_rows, filtered_df, filename)
def add_prawn_detections(sample, matching_rows, filtered_df, filename):
    """
    Add prawn detections based on bounding box information and segmentations (minimum enclosing circles).
    """
    true_detections = []
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
        largest_bbox = max(bounding_boxes, key=calculate_bbox_area)
        
        # Normalize the largest bounding box coordinates
        prawn_normalized_bbox = [
            largest_bbox[0] / 5312, largest_bbox[1] / 2988,
            largest_bbox[2] / 5312, largest_bbox[3] / 2988
        ]

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


        #diagonal line 1  of prawn_noramalized_bbox as fo polyline



        #draw    
        # Add the largest bounding box as a polyline    



        # Convert bounding box to normalized coordinates
        

        # Add true prawn detection based on the bounding box
        # true_detections.append(fo.Detection(label="prawn_true", bounding_box=prawn_normalized_bbox))

    

        # Find the closest segmentation circle to the bounding box
        segmentation = find_closest_circle_center(largest_bbox, sample["segmentations"]['polylines'])  # segmentations should be part of the sample

        # Process the prawn detection using the circle's diameter
        if segmentation is not None:
            process_detection_by_circle(segmentation, sample, filename, prawn_id, filtered_df)

    # Store true detections in the sample
    # sample["true_detections"] = fo.Detections(detections=true_detections)
    sample["diagonal_line_1"] = fo.Polylines(polylines=diagonal_line_1)
    sample["diagonal_line_2"] = fo.Polylines(polylines=diagonal_line_2)

def calculate_mean_bbox(bounding_boxes):
    """
    Calculate the mean bounding box from a list of bounding boxes.
    Each bounding box is in the format (x_min, y_min, x_max, y_max).
    """
    # Unzip the list of bounding boxes into separate lists for x_min, y_min, x_max, and y_max
    x_mins, y_mins, x_maxs, y_maxs = zip(*bounding_boxes)
    
    # Calculate the mean for each coordinate
    mean_bbox = (
        sum(x_mins) / len(x_mins),
        sum(y_mins) / len(y_mins),
        sum(x_maxs) / len(x_maxs),
        sum(y_maxs) / len(y_maxs)
    )
    
    return mean_bbox

# Convert pixel length to real-world length
# Use camera parameters and scaling as described earlier
