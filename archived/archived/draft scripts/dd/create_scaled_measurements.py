import pandas as pd
import os
import glob
import re
import numpy as np

# Create output directory if it doesn't exist
os.makedirs('fifty_one/processed_data', exist_ok=True)

# Read the scales file
scales_df = pd.read_csv('fifty_one/re (2)/Scales.csv')

# Clean up the Label column in scales_df
scales_df['Label'] = scales_df['Label'].str.replace('scales:cropped_', '')
scales_df['Label'] = scales_df['Label'].str.replace('_gamma 2', '_gamma')
scales_df['Label'] = scales_df['Label'].str.replace('_gamma.jpg_gamma', '_gamma')
scales_df['Label'] = scales_df['Label'].str.replace('.jpg', '')

# Calculate scale factor (1/(Length/10))
scales_df['scale_factor'] = 1 / (scales_df['Length'] / 10)

# Print the first few rows of the scales dataframe
print("Scales dataframe:")
print(scales_df[['Label', 'Length', 'scale_factor']].head())

# Find all measurement files
measurement_files = glob.glob('fifty_one/re (2)/meausrements-fixed-total*.csv') 
print(f"\nFound {len(measurement_files)} measurement files")

# Function to extract base image name from the full image name
def get_base_image_name(full_name):
    # Handle non-string values
    if not isinstance(full_name, str):
        return None
        
    # Extract the part before cropped coordinates
    match = re.search(r'(.*?)_gamma_obj\d+_cropped', full_name)
    if match:
        base_name = match.group(1)
        return base_name + '_gamma'
    return None

# Function to extract camera and frame from image name
def get_camera_frame(name):
    if not isinstance(name, str):
        return None
        
    match = re.search(r'(GX\d+_\d+_\d+)', name)
    if match:
        return match.group(1)
    return None

# Combined dataframe to store all scaled measurements
all_data = []

# Process each measurement file
for file_path in measurement_files:
    print(f"\nProcessing {os.path.basename(file_path)}")
    
    # Extract measurement type from filename (carapace or total)
    file_name = os.path.basename(file_path)
    if 'carapace' in file_name.lower():
        meas_type = 'carapace'
    elif 'total' in file_name.lower():
        meas_type = 'total'
    else:
        meas_type = 'unknown'
        
    # Read the measurement file
    try:
        meas_df = pd.read_csv(file_path)
        print(f"  Loaded {len(meas_df)} measurements")
    except Exception as e:
        print(f"  Error reading file: {e}")
        continue
    
    # Check if image_name column exists
    if 'image_name' not in meas_df.columns:
        print(f"  Error: 'image_name' column not found in {file_path}")
        print(f"  Available columns: {meas_df.columns.tolist()}")
        continue
    
    # Add base image name column for joining with scales
    meas_df['base_image_name'] = meas_df['image_name'].apply(get_base_image_name)
    
    # Join with scales dataframe
    merged_df = pd.merge(
        meas_df, 
        scales_df[['Label', 'scale_factor']], 
        left_on='base_image_name', 
        right_on='Label', 
        how='left'
    )
    
    # Count matches
    match_count = merged_df['scale_factor'].notna().sum()
    print(f"  Matched {match_count} of {len(meas_df)} measurements with scales")
    
    if match_count == 0:
        print("  No matches found, trying alternative matching approach")
        # Use the extracted camera ID and frame number
        meas_df['camera_frame'] = meas_df['image_name'].apply(get_camera_frame)
        scales_df['camera_frame'] = scales_df['Label'].apply(get_camera_frame)
        
        merged_df = pd.merge(
            meas_df, 
            scales_df[['camera_frame', 'scale_factor']], 
            on='camera_frame', 
            how='left'
        )
        
        match_count = merged_df['scale_factor'].notna().sum()
        print(f"  Alternative matching found {match_count} of {len(meas_df)} measurements with scales")
    
    if match_count > 0:
        # Apply scaling to measurements
        for col in ['meas_1', 'meas_2', 'meas_3', 'avg']:
            if col in merged_df.columns:
                merged_df[f'scaled_{col}'] = merged_df[col] * merged_df['scale_factor']
        
        # Add measurement type
        merged_df['measurement_type'] = meas_type
        
        # Add to combined data
        all_data.append(merged_df)
    else:
        print("  Skipping file due to no scale matches")

# Combine all data
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save the combined data
    output_path = 'fifty_one/processed_data/scaled_measurements_total.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\nSaved combined dataset with {len(combined_df)} rows to {output_path}")
    
    # Display summary
    print("\nSummary statistics for scaled measurements:")
    for meas_type in combined_df['measurement_type'].unique():
        subset = combined_df[combined_df['measurement_type'] == meas_type]
        print(f"\n{meas_type.capitalize()} measurements:")
        print(f"  Count: {len(subset)}")
        for col in ['scaled_meas_1', 'scaled_meas_2', 'scaled_meas_3', 'scaled_avg']:
            if col in subset.columns:
                print(f"  {col}: mean = {subset[col].mean():.2f}, std = {subset[col].std():.2f}")
else:
    print("No data was processed successfully") 