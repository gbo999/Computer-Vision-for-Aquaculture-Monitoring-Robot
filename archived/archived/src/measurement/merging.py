import pandas as pd

# Load the three Excel files into DataFrames
file_1_path = r"C:\Users\gbo10\OneDrive\measurement_paper_images\compile 3 files/1_Carapace.xlsx"
file_2_path = r"C:\Users\gbo10\OneDrive\measurement_paper_images\compile 3 files/2_Carapace.xlsx"
file_3_path = r"C:\Users\gbo10\OneDrive\measurement_paper_images\compile 3 files/3_Carapace.xlsx"

data_1 = pd.read_excel(file_1_path)
data_2 = pd.read_excel(file_2_path)
data_3 = pd.read_excel(file_3_path)

# Function to extract the pixel-to-mm scale from the second row of each image
def extract_scale(data):
    return data.groupby('Label').apply(lambda x: x.iloc[1]['Length'] / 10).reset_index(name='Scale')

# Extract scales for each dataset
scale_1 = extract_scale(data_1)
scale_2 = extract_scale(data_2)
scale_3 = extract_scale(data_3)

# Function to adjust bounding box coordinates based on the pixel-to-mm scale
def adjust_bounding_box(data, scale):
    data = data.copy()
    prawn_data = data.groupby('Label').apply(lambda x: x.iloc[2:]).reset_index(drop=True)  # Skip the first two rows
    prawn_data['BX'] *= scale
    prawn_data['BY'] *= scale
    prawn_data['Width'] *= scale
    prawn_data['Height'] *= scale
    return prawn_data

# Adjust bounding boxes in each dataset for each image separately
def process_image(image_label, data_1, data_2, data_3, scale_1, scale_2, scale_3, tolerance):
    img_data_1 = data_1[data_1['Label'] == image_label]
    img_data_2 = data_2[data_2['Label'] == image_label]
    img_data_3 = data_3[data_3['Label'] == image_label]

    scale_val_1 = scale_1[scale_1['Label'] == image_label]['Scale'].values[0]
    scale_val_2 = scale_2[scale_2['Label'] == image_label]['Scale'].values[0]
    scale_val_3 = scale_3[scale_3['Label'] == image_label]['Scale'].values[0]

    prawn_data_1 = adjust_bounding_box(img_data_1, scale_val_1)
    prawn_data_2 = adjust_bounding_box(img_data_2, scale_val_2)
    prawn_data_3 = adjust_bounding_box(img_data_3, scale_val_3)

    # Assign PrawnID by comparing bounding boxes with tolerance within the same image
    prawn_data_1['PrawnID'] = None
    prawn_data_2['PrawnID'] = None
    prawn_data_3['PrawnID'] = None

    for i, row1 in prawn_data_1.iterrows():
        bx1, by1, width1, height1 = row1['BX'], row1['BY'], row1['Width'], row1['Height']
        area1 = width1 * height1

        for j, row2 in prawn_data_2.iterrows():
            bx2, by2, width2, height2 = row2['BX'], row2['BY'], row2['Width'], row2['Height']
            area2 = width2 * height2

            if abs(area1 - area2) <= (tolerance * area1) and abs(bx1 - bx2) <= tolerance and abs(by1 - by2) <= tolerance:
                prawn_data_1.at[i, 'PrawnID'] = f"{image_label}_{i}"
                prawn_data_2.at[j, 'PrawnID'] = f"{image_label}_{i}"
                break

        for k, row3 in prawn_data_3.iterrows():
            bx3, by3, width3, height3 = row3['BX'], row3['BY'], row3['Width'], row3['Height']
            area3 = width3 * height3

            if abs(area1 - area3) <= (tolerance * area1) and abs(bx1 - bx3) <= tolerance and abs(by1 - by3) <= tolerance:
                prawn_data_1.at[i, 'PrawnID'] = f"{image_label}_{i}"
                prawn_data_3.at[k, 'PrawnID'] = f"{image_label}_{i}"
                break

    return prawn_data_1, prawn_data_2, prawn_data_3

# Set tolerance for bounding box comparison
tolerance = 15 # Adjust based on acceptable variance

# Process each image separately
processed_data_1 = []
processed_data_2 = []
processed_data_3 = []

for image_label in data_1['Label'].unique():
    img_data_1, img_data_2, img_data_3 = process_image(image_label, data_1, data_2, data_3, scale_1, scale_2, scale_3, tolerance)
    processed_data_1.append(img_data_1)
    processed_data_2.append(img_data_2)
    processed_data_3.append(img_data_3)

# Combine the processed data into DataFrames
final_data_1 = pd.concat(processed_data_1, ignore_index=True)
final_data_2 = pd.concat(processed_data_2, ignore_index=True)
final_data_3 = pd.concat(processed_data_3, ignore_index=True)

# Merge datasets on PrawnID and Label to combine the measurements across the three files
# Merge datasets on PrawnID and Label to combine the measurements across the three files
merged_data = final_data_1.merge(final_data_2, on=['Label', 'PrawnID'], suffixes=('_1', '_2'))

# Rename the columns in final_data_3 to ensure they have a unique suffix
final_data_3 = final_data_3.rename(columns={
    'SN': 'SN_3',
    'Area': 'Area_3',
    'Mean': 'Mean_3',
    'Min': 'Min_3',
    'Max': 'Max_3',
    'BX': 'BX_3',
    'BY': 'BY_3',
    'Width': 'Width_3',
    'Height': 'Height_3',
    'Angle': 'Angle_3',
    'Slice': 'Slice_3',
    'Length': 'Length_3'
})


merged_data = merged_data.merge(final_data_3, on=['Label', 'PrawnID'])
# Check if the length columns are present
print(merged_data.columns)


# Calculate statistics for each prawn
def calculate_statistics(merged_data):
    # Calculate mean, standard deviation, and uncertainty
    merged_data['Mean_Length'] = merged_data[['Length_1', 'Length_2', 'Length_3']].mean(axis=1)
    merged_data['Std_Dev_Length'] = merged_data[['Length_1', 'Length_2', 'Length_3']].std(axis=1)
    merged_data['Uncertainty'] = merged_data['Std_Dev_Length'] / (3 ** 0.5)
    return merged_data

# Apply the statistics calculation
final_data = calculate_statistics(merged_data)

# Concat the bounding box and angle information for verification
final_data['BoundingBox_1'] = final_data[['BX_1', 'BY_1', 'Width_1', 'Height_1']].apply(tuple, axis=1)
final_data['BoundingBox_2'] = final_data[['BX_2', 'BY_2', 'Width_2', 'Height_2']].apply(tuple, axis=1)
final_data['BoundingBox_3'] = final_data[['BX_3', 'BY_3', 'Width_3', 'Height_3']].apply(tuple, axis=1)

final_data['Angle_1'] = final_data['Angle_1']
final_data['Angle_2'] = final_data['Angle_2']
final_data['Angle_3'] = final_data['Angle_3']

# Display the final dataframe with PrawnID, calculated statistics, bounding boxes, and angles
final_data[['Label', 'PrawnID', 'Mean_Length', 'Std_Dev_Length', 'Uncertainty', 
            'BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3', 
            'Angle_1', 'Angle_2', 'Angle_3']]
#save the final data to a new Excel file
final_data.to_excel("final_data.xlsx", index=False)
# Display the final dataframe with PrawnID, calculated statistics, bounding boxes, and angles


import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tempfile

# Load the three Excel files into DataFrames
file_1_path = r"C:\Users\gbo10\OneDrive\measurement_paper_images\compile 3 files/1_Carapace.xlsx"
file_2_path = r"C:\Users\gbo10\OneDrive\measurement_paper_images\compile 3 files/2_Carapace.xlsx"
file_3_path = r"C:\Users\gbo10\OneDrive\measurement_paper_images\compile 3 files/3_Carapace.xlsx"

data_1 = pd.read_excel(file_1_path)
data_2 = pd.read_excel(file_2_path)
data_3 = pd.read_excel(file_3_path)

# Function to extract the pixel-to-mm scale from the second row of each image
def extract_scale(data):
    return data.groupby('Label').apply(lambda x: x.iloc[1]['Length'] / 10).reset_index(name='Scale')

# Extract scales for each dataset
scale_1 = extract_scale(data_1)
scale_2 = extract_scale(data_2)
scale_3 = extract_scale(data_3)

# Function to adjust bounding box coordinates based on the pixel-to-mm scale
def adjust_bounding_box(data, scale):
    data = data.copy()
    data['BX'] *= scale
    data['BY'] *= scale
    data['Width'] *= scale
    data['Height'] *= scale
    return data

# Function to draw bounding boxes and lines
def draw_bounding_boxes_and_lines(image, prawn_data, color, thickness=2):
    bx = int(prawn_data['BX'])
    by = int(prawn_data['BY'])
    width = int(prawn_data['Width'])
    height = int(prawn_data['Height'])
    angle = prawn_data['Angle']

    if angle >= 0 and angle < 45 or angle >= 135 and angle <= 180:
        start_point = (bx, by + height)
        end_point = (bx + width, by)
    else:
        start_point = (bx, by)
        end_point = (bx + width, by + height)

    # Draw the rectangle (bounding box)
    image = cv2.rectangle(image, (bx, by), (bx + width, by + height), color, thickness)
    
    # Draw the line
    image = cv2.line(image, start_point, end_point, color, thickness)

    return image

# Function to find the closest matching bounding box based on location
def find_closest_bbox(bbox, prawn_data):
    """Finds the closest matching bounding box in prawn_data to the given bbox."""
    bx1, by1, width1, height1 = bbox

    def distance(row):
        bx2, by2, width2, height2 = row['BX'], row['BY'], row['Width'], row['Height']
        return np.sqrt((bx1 - bx2)**2 + (by1 - by2)**2)

    closest_row = prawn_data.iloc[prawn_data.apply(distance, axis=1).argmin()]
    return closest_row

# Function to manually assign PrawnIDs by visual inspection
def assign_prawn_ids_manually(image_label, data_1, data_2, data_3, scale_1, scale_2, scale_3):
    img_data_1 = data_1[data_1['Label'] == image_label]
    img_data_2 = data_2[data_2['Label'] == image_label]
    img_data_3 = data_3[data_3['Label'] == image_label]

    scale_val_1 = scale_1[scale_1['Label'] == image_label]['Scale'].values[0]
    scale_val_2 = scale_2[scale_2['Label'] == image_label]['Scale'].values[0]
    scale_val_3 = scale_3[scale_3['Label'] == image_label]['Scale'].values[0]

    # Adjust the bounding boxes according to the scale
    img_data_1 = adjust_bounding_box(img_data_1, scale_val_1)
    img_data_2 = adjust_bounding_box(img_data_2, scale_val_2)
    img_data_3 = adjust_bounding_box(img_data_3, scale_val_3)

    prawn_ids = []
    prawn_id_counter =0

    for i, row1 in img_data_1.iloc[2:].iterrows():
        print(f"Assigning ID for prawn {prawn_id_counter+1} in image {image_label}")


        file_name = image_label.split(":")[1] if ":" in image_label else image_label

        file_root, file_ext = os.path.splitext(file_name)
      
    # Append a .jpg extension if it’s missing
      

        file_name = f"{file_root}{file_ext}.jpg"

       # Load the image (assuming you have a way to load it based on the label)
        image_path = f"C:/Users/gbo10/OneDrive/measurement_paper_images/images used for imageJ/check/stabilized/shai/measurements/1/carapace/{file_name}"  
        
        print(image_path)   
        image = cv2.imread(image_path)

        # Draw bounding boxes and lines from dataset 1
        image = draw_bounding_boxes_and_lines(image, row1, (255, 0, 0))  # Blue for file 1
        
        # Find the closest matching bounding boxes in the other datasets
        closest_bbox_2 = find_closest_bbox((row1['BX'], row1['BY'], row1['Width'], row1['Height']), img_data_2)
        closest_bbox_3 = find_closest_bbox((row1['BX'], row1['BY'], row1['Width'], row1['Height']), img_data_3)

        # Draw bounding boxes and lines from dataset 2 and 3
        image = draw_bounding_boxes_and_lines(image, closest_bbox_2, (0, 255, 0))  # Green for file 2
        image = draw_bounding_boxes_and_lines(image, closest_bbox_3, (0, 0, 255))  # Red for file 3


        # Display the image using Matplotlib
    

        # Manually input the ID
        prawn_id = f"Prawn_{prawn_id_counter}"

      


        prawn_id_counter += 1
     

        # Assign PrawnID to closest matches in all datasets
        img_data_1.at[i, 'PrawnID'] = prawn_id
        img_data_2.at[closest_bbox_2.name, 'PrawnID'] = prawn_id
        img_data_3.at[closest_bbox_3.name, 'PrawnID'] = prawn_id

    return img_data_1, img_data_2, img_data_3

# Process each image and assign PrawnIDs manually
processed_data_1 = []
processed_data_2 = []
processed_data_3 = []

for image_label in data_1['Label'].unique():
    img_data_1, img_data_2, img_data_3 = assign_prawn_ids_manually(image_label, data_1, data_2, data_3, scale_1, scale_2, scale_3)
    processed_data_1.append(img_data_1)
    processed_data_2.append(img_data_2)
    processed_data_3.append(img_data_3)

# Combine the processed data into DataFrames
final_data_1 = pd.concat(processed_data_1, ignore_index=True)
final_data_2 = pd.concat(processed_data_2, ignore_index=True)
final_data_3 = pd.concat(processed_data_3, ignore_index=True)

# Save the final data with manually assigned PrawnIDs
final_data_1.to_excel("final_data_1_with_prawn_ids.xlsx", index=False)
final_data_2.to_excel("final_data_2_with_prawn_ids.xlsx", index=False)
final_data_3.to_excel("final_data_3_with_prawn_ids.xlsx", index=False)
