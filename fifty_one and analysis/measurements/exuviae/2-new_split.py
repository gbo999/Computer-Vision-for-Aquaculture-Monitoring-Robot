import pandas as pd
import numpy as np
import os

# df_comn=pd.read_csv('combined_pond.csv')

df_analysis=pd.read_csv(r'training and val output/runs/pose/predict83/length_analysis_new.csv')


new_rows = []

# Iterate through each row in the original dataframe
for _, row in df_analysis.iterrows():
    image_name = row['image_name']
    
    # Create a row for big lobster if data exists
    if not pd.isna(row['big_total_length']) or not pd.isna(row['big_carapace_length']) or not pd.isna(row['Big_pixels_total_length']):
        big_row = {
            'image_name': image_name,
            'lobster_size': 'big',
            'total_length': row['big_total_length'],
            'carapace_length': row['big_carapace_length'],
            'eye_x': row['big_eye_x'],
            'eye_y': row['big_eye_y'],
            'pixels_total_length': row['Big_pixels_total_length'],
            'pixels_carapace_length': row['Big_pixels_carapace_length']
        }
        new_rows.append(big_row)
    
    # Create a row for small lobster if data exists
    if not pd.isna(row['small_total_length']) or not pd.isna(row['small_carapace_length']) or not pd.isna(row['Small_pixels_total_length']):
        small_row = {
            'image_name': image_name,
            'lobster_size': 'small',
            'total_length': row['small_total_length'],
            'carapace_length': row['small_carapace_length'],
            'eye_x': row['small_eye_x'],
            'eye_y': row['small_eye_y'],
            'pixels_total_length': row['Small_pixels_total_length'],
            'pixels_carapace_length': row['Small_pixels_carapace_length']
        }
        new_rows.append(small_row)

# Create the new dataframe
df_analysis = pd.DataFrame(new_rows)

# Sort by image_name and lobster_size
df_analysis = df_analysis.sort_values(['image_name', 'lobster_size']).reset_index(drop=True)
df_analysis.to_csv('training and val output/runs/pose/predict83/length_analysis_new_split.csv',index=False)
# Display the first few rows
