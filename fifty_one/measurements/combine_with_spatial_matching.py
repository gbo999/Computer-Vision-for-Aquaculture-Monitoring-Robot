import pandas as pd
import os
import re
import numpy as np

# Set paths
scaled_measurements_path = 'fifty_one/processed_data/scaled_measurements.csv'
body_length_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/updated_filtered_data_with_lengths_body-all.xlsx'
carapace_length_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/updated_filtered_data_with_lengths_carapace-all.xlsx'
output_dir = 'fifty_one/processed_data'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to extract center coordinates from cropped image name format
def extract_center_from_name(image_name):
    if not isinstance(image_name, str):
        return None, None
        
    if '_gamma_obj0_cropped' in image_name or '_gamma_obj5_cropped' in image_name:
        obj_cropped_pattern = r'_gamma_obj\d+_cropped([\d\.]+)_([\d\.]+)'
        match = re.search(obj_cropped_pattern, image_name)
        if match:
            try:
                center_x = float(match.group(1)) * 5312  # Image width
                center_y = float(match.group(2)) * 2988  # Image height
                return center_x, center_y
            except (IndexError, ValueError):
                pass
    return None, None

# Function to extract bounding box from format with bounding_box=()
def extract_bbox_from_name(image_name):
    if not isinstance(image_name, str):
        return None, None, None, None
        
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

# Function to extract camera frame from image name
def extract_camera_frame(image_name):
    if not isinstance(image_name, str):
        return None
    
    match = re.search(r'(GX\d+_\d+_\d+)', image_name)
    if match:
        return match.group(1)
    return None

# Function to find image ID column in a dataframe
def find_image_id_column(df):
    for col in df.columns:
        if any(isinstance(val, str) and 'GX' in str(val) for val in df[col].dropna()):
            return col
    return None

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
    print("Starting spatial merge process...")
    
    # Reset index to ensure continuous indices
    df_centers_copy = df_centers.copy().reset_index(drop=True)
    df_bboxes_copy = df_bboxes.copy().reset_index(drop=True)
    
    # Extract center coordinates from image names in df_centers
    print("Extracting center coordinates...")
    centers_data = []
    for idx, row in df_centers_copy.iterrows():
        if 'image_name' in row:
            center_x, center_y = extract_center_from_name(row['image_name'])
            image_base = None
            
            if '_gamma_obj' in str(row['image_name']):
                parts = str(row['image_name']).split('_gamma_obj')
                if len(parts) > 0:
                    image_base = parts[0] + '_gamma'
            
            # If we couldn't extract image_base, get the camera frame at least
            if not image_base:
                image_base = extract_camera_frame(row['image_name'])
                
            centers_data.append({
                'index': idx,
                'image_base': image_base,
                'center_x': center_x,
                'center_y': center_y,
                'camera_frame': extract_camera_frame(row['image_name'])
            })
    
    centers_lookup = pd.DataFrame(centers_data)
    print(f"Extracted {len(centers_lookup)} center coordinates")
    
    # Extract bounding boxes from image names in df_bboxes
    print("Extracting bounding boxes...")
    bbox_data = []
    image_id_column = find_image_id_column(df_bboxes_copy)
    
    if not image_id_column:
        print("Could not find image identifier column in bounding box dataframe")
        return None
        
    for idx, row in df_bboxes_copy.iterrows():
        image_name = row[image_id_column]
        bbox_x, bbox_y, bbox_w, bbox_h = extract_bbox_from_name(image_name)
        
        image_base = None
        if 'bounding_box=' in str(image_name):
            parts = str(image_name).split('bounding_box=')
            if len(parts) > 0:
                image_base = parts[0].strip()
        
        # If we couldn't extract image_base, get the camera frame at least
        if not image_base:
            image_base = extract_camera_frame(image_name)
            
        bbox_data.append({
            'index': idx,
            'image_base': image_base,
            'bbox_x': bbox_x,
            'bbox_y': bbox_y,
            'bbox_w': bbox_w,
            'bbox_h': bbox_h,
            'camera_frame': extract_camera_frame(image_name)
        })
    
    bbox_lookup = pd.DataFrame(bbox_data)
    print(f"Extracted {len(bbox_lookup)} bounding boxes")
    
    # First try matching by camera frame and spatial location
    match_count = 0
    matches = []
    
    for _, center_row in centers_lookup.iterrows():
        center_idx = center_row['index']
        center_x = center_row['center_x']
        center_y = center_row['center_y']
        camera_frame = center_row['camera_frame']
        
        if camera_frame is None or center_x is None or center_y is None:
            continue
            
        # Find matching bboxes by camera frame
        matching_bboxes = bbox_lookup[bbox_lookup['camera_frame'] == camera_frame]
        
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
                match_count += 1
                break  # Found a match for this center, move to next
    
    print(f"Found {match_count} matches by spatial location")
    
    # If not enough matches, try matching just by camera frame
    if match_count < len(df_centers_copy) * 0.2:  # If less than 20% matched
        print("Few spatial matches found, trying to match by camera frame only...")
        for _, center_row in centers_lookup.iterrows():
            center_idx = center_row['index']
            camera_frame = center_row['camera_frame']
            
            # Skip if already matched or no camera frame
            if any(m['center_idx'] == center_idx for m in matches) or camera_frame is None:
                continue
                
            # Find matching bboxes by camera frame
            matching_bboxes = bbox_lookup[bbox_lookup['camera_frame'] == camera_frame]
            
            if len(matching_bboxes) == 1:  # Only match if there's exactly one match to avoid ambiguity
                bbox_idx = matching_bboxes.iloc[0]['index']
                matches.append({
                    'center_idx': center_idx,
                    'bbox_idx': bbox_idx
                })
    
    print(f"Total matches found: {len(matches)}")
    
    # Create merged dataframe from matches
    if not matches:
        print("No matches found")
        return None
    
    # Verify that all indices are within range
    valid_matches = []
    for match in matches:
        center_idx = match['center_idx']
        bbox_idx = match['bbox_idx']
        
        if center_idx < 0 or center_idx >= len(df_centers_copy) or bbox_idx < 0 or bbox_idx >= len(df_bboxes_copy):
            print(f"Invalid indices: center_idx={center_idx}, bbox_idx={bbox_idx}")
            continue
            
        valid_matches.append(match)
    
    print(f"{len(valid_matches)} valid matches after index validation")
        
    merged_rows = []
    for match in valid_matches:
        center_idx = match['center_idx']
        bbox_idx = match['bbox_idx']
        
        center_row = df_centers_copy.iloc[center_idx].copy()
        bbox_row = df_bboxes_copy.iloc[bbox_idx].copy()
        
        # Create new row by combining both rows
        merged_row = {}
        
        # Add all fields from center dataframe with prefix
        for col in center_row.index:
            merged_row[f'center_{col}'] = center_row[col]
            
        # Add all fields from bbox dataframe with prefix
        for col in bbox_row.index:
            merged_row[f'bbox_{col}'] = bbox_row[col]
        
        merged_rows.append(merged_row)
    
    return pd.DataFrame(merged_rows)

# Function to match by camera frame only as fallback
def match_by_camera_frame(df_measurements, df_lengths):
    print("Matching datasets by camera frame only...")
    
    # Reset indices 
    df_measurements = df_measurements.reset_index(drop=True)
    df_lengths = df_lengths.reset_index(drop=True)
    
    # Add camera_frame to measurements dataframe if not present
    if 'camera_frame' not in df_measurements.columns:
        df_measurements['camera_frame'] = df_measurements['image_name'].apply(extract_camera_frame)
    
    # Find image ID column in length dataframe
    image_id_col = find_image_id_column(df_lengths)
    if not image_id_col:
        print("Could not find image identifier column in length dataframe")
        return None
    
    # Add camera_frame to length dataframe
    df_lengths['camera_frame'] = df_lengths[image_id_col].apply(extract_camera_frame)
    
    # Merge on camera_frame
    merged_df = pd.merge(
        df_measurements, 
        df_lengths,
        on='camera_frame',
        how='inner',
        suffixes=('_center', '_bbox')
    )
    
    print(f"Matched {len(merged_df)} records by camera frame")
    return merged_df

# Load scaled measurements data
print(f"Loading scaled measurements from {scaled_measurements_path}")
scaled_df = pd.read_csv(scaled_measurements_path)
print(f"Loaded {len(scaled_df)} scaled measurements")

# Split by measurement type
total_df = scaled_df[scaled_df['measurement_type'] == 'total'].copy()
carapace_df = scaled_df[scaled_df['measurement_type'] == 'carapace'].copy()

print(f"Total (body) measurements: {len(total_df)}")
print(f"Carapace measurements: {len(carapace_df)}")

# Load length data
print(f"\nLoading body length data from {body_length_path}")
try:
    body_length_df = pd.read_excel(body_length_path)
    print(f"Loaded {len(body_length_df)} body length records")
except Exception as e:
    print(f"Error loading body length file: {e}")
    body_length_df = None

print(f"\nLoading carapace length data from {carapace_length_path}")
try:
    carapace_length_df = pd.read_excel(carapace_length_path)
    print(f"Loaded {len(carapace_length_df)} carapace length records")
except Exception as e:
    print(f"Error loading carapace length file: {e}")
    carapace_length_df = None

# Process body/total measurements
if body_length_df is not None:
    print("\n--- Processing body (total) measurements ---")
    body_merged = merge_by_image_and_location(total_df, body_length_df)
    
    # If spatial matching fails, fall back to simpler camera frame matching
    if body_merged is None or len(body_merged) == 0:
        print("Spatial matching failed, falling back to camera frame matching")
        body_merged = match_by_camera_frame(total_df, body_length_df)
    
    if body_merged is not None:
        # Save the merged dataset
        body_output_path = os.path.join(output_dir, 'combined_body_length_data.xlsx')
        print(f"\nSaving merged body dataset to {body_output_path}")
        body_merged.to_excel(body_output_path, index=False)
        
        # Display summary statistics
        print(f"\nSummary statistics for {len(body_merged)} body records:")
        # Find the scaled measurement columns
        scaled_cols = [col for col in body_merged.columns if col.startswith('center_scaled_') or col.startswith('scaled_')]
        if scaled_cols:
            print("\nScaled measurements:")
            for col in scaled_cols:
                print(f"  {col}: mean = {body_merged[col].mean():.2f}, std = {body_merged[col].std():.2f}")
        
        # Find the length columns
        length_cols = [col for col in body_merged.columns if 'length' in col.lower() and col not in scaled_cols]
        if length_cols:
            print("\nLength measurements:")
            for col in length_cols:
                print(f"  {col}: mean = {body_merged[col].mean():.2f}, std = {body_merged[col].std():.2f}")
                
        # Calculate correlations
        if scaled_cols and length_cols:
            print("\nCorrelations:")
            for scaled_col in scaled_cols:
                for length_col in length_cols:
                    try:
                        corr = body_merged[scaled_col].corr(body_merged[length_col])
                        print(f"  {scaled_col} vs {length_col}: {corr:.4f}")
                    except:
                        pass  # Skip if correlation fails

# Process carapace measurements
if carapace_length_df is not None:
    print("\n--- Processing carapace measurements ---")
    carapace_merged = merge_by_image_and_location(carapace_df, carapace_length_df)
    
    # If spatial matching fails, fall back to simpler camera frame matching
    if carapace_merged is None or len(carapace_merged) == 0:
        print("Spatial matching failed, falling back to camera frame matching")
        carapace_merged = match_by_camera_frame(carapace_df, carapace_length_df)
    
    if carapace_merged is not None:
        # Save the merged dataset
        carapace_output_path = os.path.join(output_dir, 'combined_carapace_length_data.xlsx')
        print(f"\nSaving merged carapace dataset to {carapace_output_path}")
        carapace_merged.to_excel(carapace_output_path, index=False)
        
        # Display summary statistics
        print(f"\nSummary statistics for {len(carapace_merged)} carapace records:")
        # Find the scaled measurement columns
        scaled_cols = [col for col in carapace_merged.columns if col.startswith('center_scaled_') or col.startswith('scaled_')]
        if scaled_cols:
            print("\nScaled measurements:")
            for col in scaled_cols:
                print(f"  {col}: mean = {carapace_merged[col].mean():.2f}, std = {carapace_merged[col].std():.2f}")
        
        # Find the length columns
        length_cols = [col for col in carapace_merged.columns if 'length' in col.lower() and col not in scaled_cols]
        if length_cols:
            print("\nLength measurements:")
            for col in length_cols:
                print(f"  {col}: mean = {carapace_merged[col].mean():.2f}, std = {carapace_merged[col].std():.2f}")
                
        # Calculate correlations
        if scaled_cols and length_cols:
            print("\nCorrelations:")
            for scaled_col in scaled_cols:
                for length_col in length_cols:
                    try:
                        corr = carapace_merged[scaled_col].corr(carapace_merged[length_col])
                        print(f"  {scaled_col} vs {length_col}: {corr:.4f}")
                    except:
                        pass  # Skip if correlation fails

print("\nProcessing complete!") 