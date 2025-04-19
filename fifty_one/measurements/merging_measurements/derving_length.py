# image_name	prawn_id	meas_1	meas_2	meas_3	avg
'undistorted_GX010067_33_625.jpg_gamma_obj0_cropped0.600462_0.2647'	
# 1.jpg	232.7252	226.1415	224.2878	229.4334
import pandas as pd
import numpy as np



def extract_scale_factor(df):
    ten_mm_pixels=df['Length'].iloc[0]
    ten_mm_pixels=float(ten_mm_pixels)
    scale_factor=1/(ten_mm_pixels/10)
    return scale_factor

def merge_with_scale_factor(df):
    df['Label']=df['Label'].replace('cropped','').split('.')[0]
   
    


# Function to extract center coordinates from cropped image name format
def extract_center_from_name(image_name):
    if '_gamma_obj0_cropped' in image_name:
        try:
            coords = image_name.split('_gamma_obj0_cropped')[1].split('_')
            if len(coords) >= 2:
                center_x = float(coords[-2])*5312
                center_y = float(coords[-1])*2988
                return center_x, center_y
        except (IndexError, ValueError):
            pass
    return None, None

# Function to extract bounding box from format with bounding_box=()
def extract_bbox_from_name(image_name):
    if 'bounding_box=' in image_name:
        try:
            bbox_str = image_name.split('bounding_box=')[1].split(')')[0]
            bbox_values = [float(x.strip()) for x in bbox_str.split(',')]
            if len(bbox_values) == 4:
                x, y, width, height = bbox_values
                return x, y, width, height
        except (IndexError, ValueError):
            pass
    return None, None, None, None

# Function to check if a center point is inside a bounding box
def is_point_in_bbox(center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, tolerance=5):
    if center_x is None or center_y is None or bbox_x is None or bbox_y is None:
        return False
    
    # Check if the center point is within the bounding box with some tolerance
    in_x_range = bbox_x - tolerance <= center_x <= bbox_x + bbox_w + tolerance
    in_y_range = bbox_y - tolerance <= center_y <= bbox_y + bbox_h + tolerance
    return in_x_range and in_y_range

# Function to merge dataframes based on image name and spatial location
def merge_by_image_and_location(df_centers, df_bboxes, tolerance=5):
    """
    Merge two dataframes where one has center coordinates and the other has bounding boxes
    
    Parameters:
    - df_centers: DataFrame with image names containing center coordinates
    - df_bboxes: DataFrame with image names containing bounding box information
    - tolerance: Distance tolerance in pixels for matching
    
    Returns:
    - Merged DataFrame
    """
    # Create copies to avoid modifying the original dataframes
    df_centers_copy = df_centers.copy()
    df_bboxes_copy = df_bboxes.copy()
    
    # Extract center coordinates from image names in df_centers
    centers_data = []
    for idx, row in df_centers_copy.iterrows():
        center_x, center_y = extract_center_from_name(row['image_name'])
        centers_data.append({
            'index': idx,
            'image_base': row['image_name'].split('_gamma_obj0_cropped')[0] if '_gamma_obj0_cropped' in row['image_name'] else row['image_name'],
            'center_x': center_x,
            'center_y': center_y
        })
    centers_lookup = pd.DataFrame(centers_data)
    
    # Extract bounding boxes from image names in df_bboxes
    bbox_data = []
    for idx, row in df_bboxes_copy.iterrows():
        bbox_x, bbox_y, bbox_w, bbox_h = extract_bbox_from_name(row['image_name'])
        bbox_data.append({
            'index': idx,
            'image_base': row['image_name'].split('bounding_box=')[0].strip() if 'bounding_box=' in row['image_name'] else row['image_name'],
            'bbox_x': bbox_x,
            'bbox_y': bbox_y,
            'bbox_w': bbox_w,
            'bbox_h': bbox_h
        })
    bbox_lookup = pd.DataFrame(bbox_data)
    
    # Match entries by image name and spatial location
    matches = []
    for _, center_row in centers_lookup.iterrows():
        center_idx = center_row['index']
        center_x = center_row['center_x']
        center_y = center_row['center_y']
        image_base = center_row['image_base']
        
        # Find matching bboxes by image base name
        matching_bboxes = bbox_lookup[bbox_lookup['image_base'] == image_base]
        
        for _, bbox_row in matching_bboxes.iterrows():
            bbox_idx = bbox_row['index']
            bbox_x = bbox_row['bbox_x']
            bbox_y = bbox_row['bbox_y']
            bbox_w = bbox_row['bbox_w']
            bbox_h = bbox_row['bbox_h']
            
            # Check if the center point is inside this bounding box
            if is_point_in_bbox(center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, tolerance):
                matches.append({
                    'center_idx': center_idx,
                    'bbox_idx': bbox_idx
                })
    
    # Create merged dataframe from matches
    merged_rows = []
    for match in matches:
        center_row = df_centers_copy.iloc[match['center_idx']].copy()
        bbox_row = df_bboxes_copy.iloc[match['bbox_idx']].copy()
        
        # Create new row by combining both rows
        merged_row = {}
        for col in center_row.index:
            merged_row[f'center_{col}'] = center_row[col]
        for col in bbox_row.index:
            if col not in center_row.index:
                merged_row[f'bbox_{col}'] = bbox_row[col]
            else:
                merged_row[f'bbox_{col}'] = bbox_row[col]
        
        merged_rows.append(merged_row)
    
    return pd.DataFrame(merged_rows)

# Example usage
# Load your data
alreadt = pd.read_csv('length_analysis_new_split.csv')
df = pd.read_csv('length_analysis_new_split.csv')

# Test parsing on example data
text = 'undistorted_GX010067_33_625.jpg_gamma_obj0_cropped0.600462_0.2647'
center_x, center_y = extract_center_from_name(text)
print(f"Extracted center: ({center_x}, {center_y})")

bbox_text = 'image_name bounding_box=(2996.016573, 737.0050179, 159.9996606, 166.0006386)'
bbox_x, bbox_y, bbox_w, bbox_h = extract_bbox_from_name(bbox_text)
print(f"Extracted bbox: ({bbox_x}, {bbox_y}, {bbox_w}, {bbox_h})")

# Example of merging - assume one dataframe has center coords in image names and other has bounding boxes
# result_df = merge_by_image_and_location(df_with_centers, df_with_bboxes)
# print(result_df.head())






