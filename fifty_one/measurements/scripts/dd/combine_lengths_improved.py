import pandas as pd
import os
import re
import numpy as np
import glob
import argparse

# Set paths
# Use relative paths from this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../../..'))
# scaled_measurements_path = os.path.join(base_dir, 'fifty_one/processed_data/scaled_measurements.csv')

scaled_measurements_total_path =f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/scaled_measurements_total.csv'

scaled_measurements_carapace_path =f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/scaled_measurements_carapace.csv'




body_length_path = f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/updated_filtered_data_with_lengths_body-all.xlsx'
carapace_length_path = f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/updated_filtered_data_with_lengths_carapace-all.xlsx'
output_dir = f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to extract center coordinates from cropped image name format
def extract_center_from_name(image_name):
    """
    Placeholder function that doesn't try to extract coordinates from image names.
    
    The actual coordinates should be taken directly from the scaled_measurements.csv file
    which already has the coordinates and other measurements pre-calculated.
    
    Parameters:
    -----------
    image_name : str
        The image name (not used in this implementation)
        
    Returns:
    --------
    tuple of (None, None)
        Always returns None for both coordinates to indicate that
        coordinates should be taken from the dataframe, not the image name
    """
    # Always return None, None - coordinates should come from the dataframe
    return None, None

# Function to extract bounding box from format with bounding_box=()
def extract_bbox_from_name(image_name):
    if not isinstance(image_name, str):
        return None, None, None, None
    
    # First try the explicit 'bounding_box=' format
    if 'bounding_box=' in image_name:
        try:
            bbox_str = image_name.split('bounding_box=')[1].split(')')[0]
            bbox_values = [float(x.strip()) for x in bbox_str.split(',')]
            if len(bbox_values) == 4:
                x, y, width, height = bbox_values
                return x, y, width, height
        except (IndexError, ValueError) as e:
            print(f"Error parsing 'bounding_box=' in '{image_name}': {e}")
    
    # Try other common bbox patterns
    patterns = [
        r'bbox=\(([^)]+)\)',  # bbox=(x,y,w,h)
        r'bbox\s*\(\s*(\d+\.?\d*)[^\d]+(\d+\.?\d*)[^\d]+(\d+\.?\d*)[^\d]+(\d+\.?\d*)',  # bbox(x y w h)
        r'box=\(([^)]+)\)',   # box=(x,y,w,h)
        r'\((\d+\.?\d*)[^\d]+(\d+\.?\d*)[^\d]+(\d+\.?\d*)[^\d]+(\d+\.?\d*)\)'  # (x,y,w,h)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, image_name)
        if match:
            try:
                if len(match.groups()) == 1:  # single group with comma-separated values
                    values = [float(x.strip()) for x in match.group(1).split(',')]
                    if len(values) == 4:
                        return values[0], values[1], values[2], values[3]
                elif len(match.groups()) == 4:  # four separate groups
                    return float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
            except Exception as e:
                print(f"Error parsing pattern '{pattern}' in '{image_name}': {e}")
    
    # No valid bounding box found
    return None, None, None, None

# Function to check if a center point is inside a bounding box
def is_center_within_bbox(center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, tolerance=500):
    """
    Check if a point (center_x, center_y) is inside a bounding box with tolerance.

    Parameters:
        center_x (float): X coordinate of the center point.
        center_y (float): Y coordinate of the center point.
        bbox_x (float): X coordinate of the top-left corner of the bounding box.
        bbox_y (float): Y coordinate of the top-left corner of the bounding box.
        bbox_w (float): Width of the bounding box.
        bbox_h (float): Height of the bounding box.
        tolerance (float): Optional tolerance to expand the bounding box.

    Returns:
        bool: True if the center is within the (possibly expanded) bounding box, False otherwise.
    """
    if None in (center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h):
        return False

    within_x = (bbox_x - tolerance) <= center_x <= (bbox_x + bbox_w + tolerance)
    within_y = (bbox_y - tolerance) <= center_y <= (bbox_y + bbox_h + tolerance)

    return within_x and within_y

# Function to extract camera frame from image name
def extract_camera_frame(image_name):
    """
    Extract the camera frame identifier from an image filename.
    
    The function looks for patterns like 'GX010123_456_789' which identify
    specific frames from GoPro or similar camera footage.
    
    Parameters:
    -----------
    image_name : str
        The filename to extract camera frame from
        
    Returns:
    --------
    str or None
        The extracted camera frame identifier or None if not found
    """
    
    #if there is no jpg_gamma, return None
    if '.jpg_gamma' not in image_name:
        parts = image_name.split('_obj')
        if len(parts) > 1:
            return parts[0]
        else:
            print(f"No camera frame found in {image_name}")
            return None
    
    # Split by jpg_gamma if present
    parts = image_name.split('.jpg_gamma')
    image_name = parts[0] if parts else image_name
    return image_name

    # # Try to extract GX pattern in standard format (GX010123_456_789)
    # match = re.search(r'(GX\d+_\d+_\d+)', image_name)
    # if match:
    #     return match.group(1)
    
    # # Try alternate pattern with different separators (GX010123-456-789)
    # match = re.search(r'(GX\d+)[-_](\d+)[-_](\d+)', image_name)
    # if match:
    #     # Standardize format to underscore separator
    #     return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    # # No camera frame found
    # return None

# Function to extract bounding boxes from DataFrame columns
def extract_bboxes_from_columns(df, image_id_column):
    """
    Extract bounding box coordinates from dataframe columns.
    
    Each row is expected to have up to 3 bounding boxes.
    The function selects the first valid bounding box for each row.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing bounding box information
    image_id_column : str
        Name of the column containing image identifiers
        
    Returns:
    --------
    int
        Number of valid bounding boxes extracted
    """
    # Initialize bounding box columns
    df['bbox_x'] = None
    df['bbox_y'] = None
    df['bbox_w'] = None
    df['bbox_h'] = None
    
    # Look specifically for BoundingBox_1, BoundingBox_2, BoundingBox_3 columns
    # or other similar patterns that indicate multiple bounding boxes
    expected_bbox_columns = ['BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3']
    found_bbox_columns = [col for col in df.columns if col in expected_bbox_columns]
    
    # If we didn't find the expected columns, try a more general approach
    if not found_bbox_columns:
        found_bbox_columns = [col for col in df.columns if 'BoundingBox' in col or 'boundingbox' in col.lower()]
    
    if found_bbox_columns:
        print(f"Found {len(found_bbox_columns)} bounding box columns: {found_bbox_columns}")
        
        # Process each row
        for idx, row in df.iterrows():
            valid_bbox_found = False
            
            # Try each bounding box column in order
            for bbox_col in found_bbox_columns:
                if pd.isna(row[bbox_col]):
                    continue
                
                bbox_str = str(row[bbox_col])
                if '(' in bbox_str and ')' in bbox_str:
                    try:
                        # Extract values from format like (x, y, w, h)
                        bbox_str = bbox_str.strip('()').replace(' ', '')
                        bbox_values = [float(x.strip()) for x in bbox_str.split(',')]
                        if len(bbox_values) == 4:
                            df.at[idx, 'bbox_x'] = bbox_values[0]
                            df.at[idx, 'bbox_y'] = bbox_values[1]
                            df.at[idx, 'bbox_w'] = bbox_values[2]
                            df.at[idx, 'bbox_h'] = bbox_values[3]
                            valid_bbox_found = True
                            break  # Use first valid bounding box
                    except Exception as e:
                        print(f"Error parsing bounding box {bbox_str} in column {bbox_col} for row {idx}: {e}")
                        continue
            
            # If no valid bounding box found in columns, try extracting from image name
            if not valid_bbox_found:
                bbox_x, bbox_y, bbox_w, bbox_h = extract_bbox_from_name(str(row[image_id_column]))
                if bbox_x is not None and bbox_y is not None and bbox_w is not None and bbox_h is not None:
                    df.at[idx, 'bbox_x'] = bbox_x
                    df.at[idx, 'bbox_y'] = bbox_y
                    df.at[idx, 'bbox_w'] = bbox_w
                    df.at[idx, 'bbox_h'] = bbox_h
    else:
        # Fall back to extracting from image name if no BoundingBox columns
        print("No BoundingBox columns found, extracting from image names...")
        for idx, row in df.iterrows():
            bbox_x, bbox_y, bbox_w, bbox_h = extract_bbox_from_name(str(row[image_id_column]))
            df.at[idx, 'bbox_x'] = bbox_x
            df.at[idx, 'bbox_y'] = bbox_y
            df.at[idx, 'bbox_w'] = bbox_w
            df.at[idx, 'bbox_h'] = bbox_h
    
    # Count and return valid extractions
    valid_bboxes = df[df['bbox_x'].notna()].shape[0]
    print(f"Successfully extracted {valid_bboxes} valid bounding boxes out of {len(df)} rows")
    return valid_bboxes

# Function to extract more specific frame info from image name
def extract_specific_frame_info(image_name):
    if not isinstance(image_name, str):
        return None
    
    # Extract full GX pattern with specific object number
    # Look for patterns like GX010179_200_3927_obj1 or similar
    # This should capture the exact frame and object ID
    match = re.search(r'(GX\d+_\d+_\d+).*?obj(\d+)', image_name)
    if match:
        frame = match.group(1)
        obj_id = match.group(2)
        return f"{frame}_obj{obj_id}"
    
    # Try to extract just the frame info if no object ID found
    match = re.search(r'(GX\d+_\d+_\d+)', image_name)
    if match:
        return match.group(1)
    
    # If not found, try alternate patterns
    match = re.search(r'(GX\d+)[-_](\d+)[-_](\d+)', image_name)
    if match:
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    
    return None

# Process both body and carapace data with spatial matching
def process_measurements(measurements_df, length_df, output_filename, measurement_type, tolerance=500):
    """
    Process measurements and match them with length data.
    
    Parameters:
        measurements_df (DataFrame): DataFrame containing measurement data
        length_df (DataFrame): DataFrame containing length data
        output_filename (str): Name of the output file
        measurement_type (str): Type of measurement ('body' or 'carapace')
        tolerance (int): Tolerance in pixels for bbox matching
    """
    print(f"\n--- Processing {measurement_type} measurements ---")
    print(f"Measurements dataframe: {len(measurements_df)} rows")
    print(f"Length dataframe: {len(length_df)} rows")
    print(f"Using tolerance: {tolerance} pixels for bbox matching")
    
    # Extract camera frame from both dataframes for initial matching
    measurements_df.dropna(subset=['image_name'], inplace=True)
    print(f"Measurements dataframe after dropping na: {len(measurements_df)} rows")
    #len unique image names
    # Print camera frame column

    measurements_df['camera_frame'] = measurements_df['image_name'].astype(str).apply(extract_camera_frame)
    # measurements_df = measurements_df.sort_values(by='camera_frame').dropna(subset=['camera_frame'])
    print(measurements_df['camera_frame'].head())

    #len unique camera frames
    print(f"Measurements unique camera frames: {len(measurements_df['camera_frame'].unique())}")
    # # Find image identifier column in length_df
    # image_id_column = None
    # for col in length_df.columns:
    #     if any(isinstance(val, str) and 'GX' in str(val) for val in length_df[col].dropna()):
    #         image_id_column = col
    #         break
    
    # if not image_id_column:
    #     print("Could not find image identifier column in length dataframe")
    #     return
        
    # Process image names by splitting on ':' and 'jpg_gamma'
    length_df['camera_frame'] = length_df['Label'].apply(
        lambda x: str(x).split(':')[1].split('.jpg_gamma')[0] if isinstance(x, str) and ':' in str(x) else x
    )
    length_df = length_df.sort_values(by='camera_frame')
    # Print camera frame column
    print(f"Length dataframe after dropping na: {len(length_df)} rows")
    print(length_df['camera_frame'].head())

    #len unique camera frames
    print(f"Length unique camera frames: {len(length_df['camera_frame'].unique())}")



    # Print camera frame column
    # print(measurements_df['camera_frame'].head())

    
    # Extract center coordinates from measurements by splitting the image name with 'cropped'
    print("Extracting center coordinates from measurement images...")
    measurements_df['center_x'] = None
    measurements_df['center_y'] = None

    for idx, row in measurements_df.iterrows():
        image_name = row['image_name']
        if isinstance(image_name, str) and 'cropped' in image_name:
            try:
                parts = image_name.split('cropped')[-1].split('_')
                center_x = float(parts[0])
                center_y = float(parts[1])
                measurements_df.at[idx, 'center_x'] = center_x
                measurements_df.at[idx, 'center_y'] = center_y
            except (IndexError, ValueError) as e:
                print(f"Error extracting center coordinates from '{image_name}': {e}")
    
    # Extract all bounding boxes from length data
    print("Extracting bounding boxes from length data...")
    
    # Find all bounding box columns
    bbox_columns = []
    for i in range(1, 4):  # Assuming up to 3 bounding boxes per row
        bbox_col = f'BoundingBox_{i}'
        if bbox_col in length_df.columns:
            bbox_columns.append(bbox_col)
    
    if not bbox_columns:
        # Try general pattern if numbered columns not found
        bbox_columns = [col for col in length_df.columns if 'BoundingBox' in col or 'boundingbox' in col.lower()]
    
    print(f"Found {len(bbox_columns)} bounding box columns: {bbox_columns}")
    
    # Count valid center coordinates
    valid_centers = measurements_df[measurements_df['center_x'].notna()].shape[0]
    print(f"Valid center coordinates: {valid_centers} of {len(measurements_df)}")
    
    # Simple, correct matching approach
    print("\nMatching: First group by image names, then use spatial matching within each image")
    matches = []
    matched_meas_indices = set()
    matched_length_indices = set()
    
    # # Get common camera frames
    # meas_frames = measurements_df['camera_frame'].dropna().unique()
    # length_frames = length_df['camera_frame'].dropna().unique()


    #mathch by camera frame
    # Find frames that are close matches but not exact
    meas_frames = set(measurements_df['camera_frame'])
    length_frames = set(length_df['camera_frame'])
    common_frames = meas_frames.intersection(length_frames)
    
    # Check for near misses by comparing frame numbers
    near_misses = []
    for mf in meas_frames:
        if mf not in common_frames:
            print(f"Measurement frame {mf} not in common frames")

            # Try to find frames that differ by only a small amount
            for lf in length_frames:
                try:
                    # Extract numeric parts and compare, handling different formats
                    mf_parts = str(mf).split('_')
                    lf_parts = str(lf).split('_')
                    
                    # Compare each part of the frame number
                    if len(mf_parts) == len(lf_parts):
                        total_diff = 0
                        for mf_part, lf_part in zip(mf_parts, lf_parts):
                            mf_num = int(''.join(filter(str.isdigit, mf_part)))
                            lf_num = int(''.join(filter(str.isdigit, lf_part)))
                            total_diff += abs(mf_num - lf_num)
                            
                        if total_diff <= 5:  # Total difference threshold across all parts
                            near_misses.append((mf, lf, total_diff))
                except (ValueError, TypeError):
                    continue
    
    # Always print something about near misses, even if none found
    if near_misses:
        # Sort by difference amount
        near_misses.sort(key=lambda x: x[2])
        print("\nFound near-miss frame matches (sorted by difference):")
        for mf, lf, diff in near_misses[:5]:  # Show first 5 examples
            print(f"Measurement frame {mf} nearly matches length frame {lf} (diff: {diff})")
    else:
        print("\nNo near-miss frame matches found")

    print(f"Common camera frames: {len(common_frames)}")
    
    # Process each camera frame
    print("\nProcessing each camera frame and applying spatial matching within frame...")
    frame_matches = 0
    spatial_matches = 0
    
    for camera_frame in sorted(common_frames):
        # Get all measurements with this camera frame
        meas_group = measurements_df[measurements_df['camera_frame'] == camera_frame]
        # Get all length entries with this camera frame
        length_group = length_df[length_df['camera_frame'] == camera_frame]
        
        # Skip if either group is empty
        if meas_group.empty or length_group.empty:
            continue
            
        frame_matches_count = 0
        
        # If there's only one measurement and one length entry, it's a direct match
        if len(meas_group) == 1 and len(length_group) == 1:
            meas_idx = meas_group.index[0]
            length_idx = length_group.index[0]
            
            matches.append((meas_idx, length_idx, "frame_match"))
            matched_meas_indices.add(meas_idx)
            matched_length_indices.add(length_idx)
            frame_matches += 1
            frame_matches_count += 1
        else:
            # Multiple measurements or length entries in this frame
            # Try to match based on spatial relationship
            for meas_idx, meas_row in meas_group.iterrows():
                if meas_idx in matched_meas_indices:
                    continue  # Skip already matched measurements
                    
                center_x = meas_row['center_x']*5312
                center_y = meas_row['center_y']*2988
                
                # Skip if we don't have center coordinates
                if pd.isna(center_x) or pd.isna(center_y):
                    continue
                
                # Look for length entries whose bounding box contains this center point
                for length_idx, length_row in length_group.iterrows():
                    if length_idx in matched_length_indices:
                        continue  # Skip already matched length entries
                    
                    # Check if the center is in ANY of the bounding boxes
                    found_match = False
                    
                    for bbox_col in bbox_columns:
                        if pd.isna(length_row[bbox_col]):
                            continue
                            
                        bbox_str = str(length_row[bbox_col])
                        if '(' in bbox_str and ')' in bbox_str:
                            try:
                                # Extract values from the bounding box string
                                bbox_str = bbox_str.strip('()').replace(' ', '')
                                bbox_values = [float(x.strip()) for x in bbox_str.split(',')]
                                if len(bbox_values) == 4:
                                    bbox_x, bbox_y, bbox_w, bbox_h = bbox_values
                                    
                                    # Check if center is in this bbox
                                    if is_center_within_bbox(center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, tolerance):
                                        found_match = True
                                        break
                            except Exception as e:
                                # Just continue if there's an error with this bbox
                                pass
                    
                    if found_match:
                        matches.append((meas_idx, length_idx, "spatial_match"))
                        matched_meas_indices.add(meas_idx)
                        matched_length_indices.add(length_idx)
                        spatial_matches += 1
                        frame_matches_count += 1
                        break
        
        # Report results for this frame
        if frame_matches_count > 0:
            print(f"  Frame {camera_frame}: {len(meas_group)} measurements, {len(length_group)} length entries, {frame_matches_count} matches")
    
    # Print match statistics
    print(f"\nFound {len(matches)} total matches")
    print(f"Frame matches (1-to-1): {frame_matches}")
    print(f"Spatial matches (within frames): {spatial_matches}")
    
    # Create merged dataframe
    if matches:
        # Get unique measurement indices
        unique_meas_indices = list(set(m[0] for m in matches))
        print(f"Unique measurement matches: {len(unique_meas_indices)}")
        
        # Create merged rows
        merged_rows = []
        
        for meas_idx, length_idx, match_type in matches:
            meas_row = measurements_df.loc[meas_idx]
            length_row = length_df.loc[length_idx]
            
            # Create merged row
            merged_row = {}
            
            # Add measurement data
            for col in measurements_df.columns:
                merged_row[f'meas_{col}'] = meas_row[col]
            
            # Add length data - only including the important columns
            for col in length_df.columns:
                # Include all columns, including bounding box columns
                merged_row[f'length_{col}'] = length_row[col]
            
            # Add match type
            merged_row['match_type'] = match_type
            
            merged_rows.append(merged_row)
        
        # Create dataframe from merged rows
        merged_df = pd.DataFrame(merged_rows)
        
        # Keep only first match for each measurement if we have duplicates
        if len(merged_df) > len(unique_meas_indices):
            print("Removing duplicate matches...")
            merged_df = merged_df.drop_duplicates(subset=['meas_image_name'])
            print(f"After removing duplicates: {len(merged_df)} matches")
        
        # Save to CSV file
        output_path = os.path.join(output_dir, output_filename)
        print(f"Saving to {output_path}")
        merged_df.to_csv(output_path, index=False)
        
        # Show statistics and correlations
        print("\nSummary statistics:")
        
        # Find scaled measurement columns
        scaled_cols = [col for col in merged_df.columns if 'scaled_' in col]
        print(f"Found {len(scaled_cols)} scaled measurement columns")
        
        # Find length columns
        length_cols = [col for col in merged_df.columns if 'length_' in col and 
                       merged_df[col].dtype in [np.float64, np.int64]]
        print(f"Found {len(length_cols)} numeric length columns")
        
        # Show correlations
        if scaled_cols and length_cols:
            print("\nTop correlations between scaled measurements and lengths:")
            correlations = []
            
            for scaled_col in scaled_cols:
                for length_col in length_cols:
                    try:
                        corr = merged_df[scaled_col].corr(merged_df[length_col])
                        if not np.isnan(corr):
                            correlations.append((scaled_col, length_col, corr))
                    except:
                        pass
            
            # Sort by absolute correlation and display top 10
            correlations.sort(key=lambda x: abs(x[2]), reverse=True)
            for scaled_col, length_col, corr in correlations[:10]:
                print(f"  {scaled_col} vs {length_col}: {corr:.4f}")
        
        # Identify and print unmatched camera frames
        unmatched_measurements_frames = set(measurements_df['camera_frame']) - common_frames
        unmatched_length_frames = set(length_df['camera_frame']) - common_frames

        print("Unmatched camera frames in measurements_df:")
        for frame in list(unmatched_measurements_frames)[:5]:  # Print first 5 examples
            print(frame)

        print("Unmatched camera frames in length_df:")
        for frame in list(unmatched_length_frames)[:5]:  # Print first 5 examples
            print(frame)
        
        return merged_df
    else:
        print("No matches found")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Combine measurements with length data')
    parser.add_argument('--tolerance', type=int, default=500, 
                        help='Tolerance in pixels for bbox matching (default: 500)')
    args = parser.parse_args()
    
    # Split by measurement type
    total_df = pd.read_csv(scaled_measurements_total_path)
    carapace_df = pd.read_csv(scaled_measurements_carapace_path)

    # Load body length data
    print(f"\nLoading body length data from {body_length_path}")
    try:
        body_length_df = pd.read_excel(body_length_path)
        print(f"Loaded {len(body_length_df)} body length records")
    except Exception as e:
        print(f"Error loading body length file: {e}")
        body_length_df = None

    # Load carapace length data
    print(f"\nLoading carapace length data from {carapace_length_path}")
    try:
        carapace_length_df = pd.read_excel(carapace_length_path)
        print(f"Loaded {len(carapace_length_df)} carapace length records")
    except Exception as e:
        print(f"Error loading carapace length file: {e}")
        carapace_length_df = None

    # Process body data
    if body_length_df is not None:
        body_result = process_measurements(total_df, body_length_df, 
                                         'combined_body_length_data.csv', 
                                         'body/total', args.tolerance)

    # Process carapace data
    if carapace_length_df is not None:
        carapace_result = process_measurements(carapace_df, carapace_length_df, 
                                             'combined_carapace_length_data.csv', 
                                             'carapace', args.tolerance)

    print("\nProcessing complete!")

