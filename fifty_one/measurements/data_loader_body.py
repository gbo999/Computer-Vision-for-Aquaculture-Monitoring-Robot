from MEC import welzl
from utils import calculate_euclidean_distance
from utils import calculate_real_width
import os
import ast
from tqdm import tqdm
import fiftyone as fo
import pandas as pd
from MEC import Point
import fiftyone.core.labels as fol
#pixels and segmentation not jsut center

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

    # Open the segmentation file and process each line
    with open(segmentation_path, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split()))  # Convert the line to a list of floats
            normalzied_points = [(coords[i]/640, coords[i + 1]/640) for i in range(0, len(coords), 2)]  # Extract points (x, y)
            points = [Point(x*5312, y*2988) for x, y in normalzied_points] 
            

            
             # Convert to Point objects    
            # Calculate the minimum enclosing circle (center and radius)
            center, radius = calculate_minimum_enclosing_circle(points)
            diameter = radius * 2
             
            segmentation = fo.Polyline(
                points=[normalzied_points],
                closed=True,
                filled=True,
                diameter=diameter,
                center=center
            )
            segmentations.append(segmentation)
            
            # Store the segmentation information (center, radius, and diameter)

    return segmentations

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
    height_mm = sample['heigtht(mm)']  # Height of the prawn
    focal_length = 24.22  # Camera focal length
    pixel_size = 0.00716844  # Pixel size in mm

    poly=segmentation[0]

    # Get the diameter of the circle in pixels
    predicted_diameter_pixels = poly['diameter']

    # Calculate the real-world prawn size using the enclosing circle's diameter
    real_length_cm = calculate_real_width(focal_length, height_mm, predicted_diameter_pixels, pixel_size)

    # Update the filtered dataframe with the calculated real length
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'RealLength(cm)'] = real_length_cm

    # Fetch the true length from the dataframe
    true_length = filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Avg_Length'].values[0]

    # Compute error and create label for the closest detection
    error_percentage = abs(real_length_cm - true_length) / true_length * 100
    closest_detection_label = f'MPError: {error_percentage:.2f}%, true length: {true_length:.2f}cm, pred length: {real_length_cm:.2f}cm'
    poly.label = closest_detection_label
    poly.attributes["prawn_id"] = fo.Attribute(value=prawn_id)
    # Attach information to the sample
    

    # Tagging the sample based on error percentage
    if error_percentage > 25:
        if "MPE>25" not in sample.tags:
            sample.tags.append("MPE>25")
    else:
        if "MPE<25" not in sample.tags:
            sample.tags.append("MPE<25")
def process_images(image_paths, prediction_folder_path, filtered_df, metadata_df, dataset):
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
        segmentations = process_segmentations(prediction_txt_path)

        # Load bounding boxes from the filtered data
        matching_rows = filtered_df[filtered_df['Label'] == f'full body:{filename}']
        
        if matching_rows.empty:
            print(f"No bounding boxes found for {filename}")
            continue

        print(image_path)    
        # Create a new sample for FiftyOne
        sample = fo.Sample(filepath=image_path)

        # Iterate over each bounding box in the filtered data
        sample["segmentations"] = fol.Polylines(polylines=segmentations)
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

    for _, row in matching_rows.iterrows():
        prawn_id = row['PrawnID']
        prawn_bbox = ast.literal_eval(row['BoundingBox_1'])

        # Convert bounding box to normalized coordinates
        prawn_bbox = tuple(float(coord) for coord in prawn_bbox)
        prawn_normalized_bbox = [prawn_bbox[0] / 5312, prawn_bbox[1] / 2988, prawn_bbox[2] / 5312, prawn_bbox[3] / 2988]

        # Add true prawn detection based on the bounding box
        true_detections.append(fo.Detection(label="prawn_true", bounding_box=prawn_normalized_bbox))

    

        # Find the closest segmentation circle to the bounding box
        segmentation = find_closest_circle_center(prawn_bbox, sample["segmentations"]['polylines'])  # segmentations should be part of the sample

        # Process the prawn detection using the circle's diameter
        if segmentation is not None:
            process_detection_by_circle(segmentation, sample, filename, prawn_id, filtered_df)

    # Store true detections in the sample
    sample["true_detections"] = fo.Detections(detections=true_detections)
