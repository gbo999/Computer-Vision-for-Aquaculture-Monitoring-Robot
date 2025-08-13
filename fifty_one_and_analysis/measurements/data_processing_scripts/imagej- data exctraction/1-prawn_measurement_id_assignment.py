"""
1- Prawn Measurement ID Assignment

This script is the first step in the prawn measurement analysis pipeline. It processes ImageJ 
measurement data from three separate datasets and assigns consistent prawn IDs across all measurements.

Processing Steps:
1. Loads measurement data from three separate ImageJ output files
2. Extracts and applies pixel-to-mm scale calibration
3. Matches corresponding prawn measurements across datasets using spatial proximity
4. Assigns consistent IDs to matched measurements
5. Saves the processed data with assigned IDs for further analysis

Input Files:
- 1_Full_body.xlsx: First set of ImageJ measurements
- 2_Full_body.xlsx: Second set of ImageJ measurements
- 3_Full_body.xlsx: Third set of ImageJ measurements

Output Files:
- final_full_data_1_with_prawn_ids.xlsx
- final_full_data_2_with_prawn_ids.xlsx
- final_full_data_3_with_prawn_ids.xlsx

Dependencies:
    - pandas: For data manipulation and Excel I/O
    - cv2: For image processing and visualization
    - numpy: For numerical computations
    - matplotlib: For visualization
    - tqdm: For progress tracking
"""

import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tempfile
from tqdm import tqdm


def extract_scale(data):
    """
    Extract pixel-to-millimeter scale from calibration measurements.
    
    The scale is calculated from the second row of each image's measurements,
    which contains a calibration measurement. The Length value is divided by 10
    to convert from pixels to millimeters.
    
    Args:
        data (pd.DataFrame): DataFrame containing ImageJ measurements
        
    Returns:
        pd.DataFrame: DataFrame with Label and Scale columns
    """
    return data.groupby('Label').apply(lambda x: x.iloc[1]['Length'] / 10).reset_index(name='Scale')


def adjust_bounding_box(data, scale):
    """
    Adjust bounding box coordinates using pixel-to-mm scale.
    
    Converts pixel coordinates to real-world measurements by applying the
    calibration scale to all spatial measurements.
    
    Args:
        data (pd.DataFrame): DataFrame containing bounding box coordinates
        scale (float): Pixel-to-millimeter conversion factor
        
    Returns:
        pd.DataFrame: DataFrame with scaled coordinates
    """
    data = data.copy()
    data['BX'] *= scale
    data['BY'] *= scale
    data['Width'] *= scale
    data['Height'] *= scale
    return data


def draw_bounding_boxes_and_lines(image, prawn_data, color, thickness=2):
    """
    Draw bounding boxes and measurement lines on the image.
    
    Visualizes the prawn measurements by drawing:
    - A rectangle around the detected prawn
    - A line indicating the measurement axis
    
    The line orientation is adjusted based on the measurement angle.
    
    Args:
        image: OpenCV image to draw on
        prawn_data (pd.Series): Row containing measurement data
        color: RGB color tuple for drawing
        thickness (int): Line thickness in pixels
        
    Returns:
        image: Modified image with drawings
    """
    bx = int(prawn_data['BX'])
    by = int(prawn_data['BY'])
    width = int(prawn_data['Width'])
    height = int(prawn_data['Height'])
    angle = prawn_data['Angle']

    # Determine line endpoints based on angle
    if angle >= 0 and angle < 45 or angle >= 135 and angle <= 180:
        start_point = (bx, by + height)
        end_point = (bx + width, by)
    else:
        start_point = (bx, by)
        end_point = (bx + width, by + height)

    # Draw rectangle and line
    image = cv2.rectangle(image, (bx, by), (bx + width, by + height), color, thickness)
    image = cv2.line(image, start_point, end_point, color, thickness)

    return image


def find_closest_bbox(bbox, prawn_data):
    """
    Find the closest matching bounding box in the dataset.
    
    Uses Euclidean distance between box centers to find the most likely
    corresponding measurement in another dataset. Ignores the first two rows
    which contain calibration data.
    
    Args:
        bbox (tuple): (bx, by, width, height) of the reference box
        prawn_data (pd.DataFrame): DataFrame containing candidate boxes
        
    Returns:
        pd.Series: Row containing the closest matching measurement
    """
    bx1, by1, width1, height1 = bbox
    prawn_data_filtered = prawn_data.iloc[2:]  # Skip calibration rows

    def distance(row):
        bx2, by2 = row['BX'], row['BY']
        return np.sqrt((bx1 - bx2)**2 + (by1 - by2)**2)

    closest_row = prawn_data_filtered.iloc[prawn_data_filtered.apply(distance, axis=1).argmin()]
    return closest_row


def assign_prawn_ids_manually(image_label, data_1, data_2, data_3, scale_1, scale_2, scale_3):
    """
    Assign consistent IDs to corresponding prawn measurements across datasets.
    
    For each image:
    1. Scales all measurements to real-world coordinates
    2. Matches corresponding measurements across datasets
    3. Assigns consistent IDs to matched measurements
    
    Args:
        image_label (str): Identifier for the current image
        data_1, data_2, data_3 (pd.DataFrame): Measurement data from each dataset
        scale_1, scale_2, scale_3 (pd.DataFrame): Scaling factors for each dataset
        
    Returns:
        tuple: (img_data_1, img_data_2, img_data_3) DataFrames with assigned IDs
    """
    # Extract data for current image
    img_data_1 = data_1[data_1['Label'] == image_label]
    img_data_2 = data_2[data_2['Label'] == image_label]
    img_data_3 = data_3[data_3['Label'] == image_label]

    # Get scaling factors
    scale_val_1 = scale_1[scale_1['Label'] == image_label]['Scale'].values[0]
    scale_val_2 = scale_2[scale_2['Label'] == image_label]['Scale'].values[0]
    scale_val_3 = scale_3[scale_3['Label'] == image_label]['Scale'].values[0]

    # Scale measurements
    img_data_1 = adjust_bounding_box(img_data_1, scale_val_1)
    img_data_2 = adjust_bounding_box(img_data_2, scale_val_2)
    img_data_3 = adjust_bounding_box(img_data_3, scale_val_3)

    prawn_id_counter = 0

    # Assign IDs to matched measurements
    for i, row1 in img_data_1.iloc[2:].iterrows():
        closest_bbox_2 = find_closest_bbox((row1['BX'], row1['BY'], row1['Width'], row1['Height']), img_data_2)
        closest_bbox_3 = find_closest_bbox((row1['BX'], row1['BY'], row1['Width'], row1['Height']), img_data_3)

        prawn_id = f"Prawn_{prawn_id_counter}"
        print(prawn_id_counter)
        prawn_id_counter += 1

        # Assign ID to matched measurements
        img_data_1.at[i, 'PrawnID'] = prawn_id
        img_data_2.at[closest_bbox_2.name, 'PrawnID'] = prawn_id
        img_data_3.at[closest_bbox_3.name, 'PrawnID'] = prawn_id

    return img_data_1, img_data_2, img_data_3


def main():
    """
    Main execution function that processes all images and assigns prawn IDs.
    """
    # Load input files
    # file_1_path = r"OneDrive\measurement_paper_images\compile 3 files/1_Full_body.xlsx"
# file_2_path = r"OneDrive\measurement_paper_images\compile 3 files/2_Full_body.xlsx"
# file_3_path = r"OneDrive\measurement_paper_images\compile 3 files/3_Full_body.xlsx"
# Use OneDrive paths for data sharing - keeping these as they're needed for collaboration
    file_1_path = r"OneDrive\measurement_paper_images\compile 3 files/1_Full_body.xlsx"
    file_2_path = r"OneDrive\measurement_paper_images\compile 3 files/2_Full_body.xlsx"
    file_3_path = r"OneDrive\measurement_paper_images\compile 3 files/3_Full_body.xlsx"

    data_1 = pd.read_excel(file_1_path)
    data_2 = pd.read_excel(file_2_path)
    data_3 = pd.read_excel(file_3_path)

    # Extract scaling factors
    scale_1 = extract_scale(data_1)
    scale_2 = extract_scale(data_2)
    scale_3 = extract_scale(data_3)

    # Process each image
    processed_data_1 = []
    processed_data_2 = []
    processed_data_3 = []

    for image_label in tqdm(data_1['Label'].unique()):
        img_data_1, img_data_2, img_data_3 = assign_prawn_ids_manually(
            image_label, data_1, data_2, data_3, scale_1, scale_2, scale_3)
        processed_data_1.append(img_data_1)
        processed_data_2.append(img_data_2)
        processed_data_3.append(img_data_3)

    # Combine and save results
    final_data_1 = pd.concat(processed_data_1, ignore_index=True)
    final_data_2 = pd.concat(processed_data_2, ignore_index=True)
    final_data_3 = pd.concat(processed_data_3, ignore_index=True)

    final_data_1.to_excel("final_full_data_1_with_prawn_ids.xlsx", index=False)
    final_data_2.to_excel("final_full_data_2_with_prawn_ids.xlsx", index=False)
    final_data_3.to_excel("final_full_data_3_with_prawn_ids.xlsx", index=False)


if __name__ == "__main__":
    main()
