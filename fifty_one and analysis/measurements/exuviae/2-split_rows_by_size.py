import pandas as pd
import numpy as np
import os

def transform_and_save_analysis_data(input_csv_path: str, output_csv_path: str) -> None:
    """
    Transforms the analysis data from a CSV file by splitting rows based on lobster size
    and saves the transformed data to a new CSV file.

    This function reads a CSV file containing analysis data of lobsters, processes each row
    to separate data for big and small lobsters, and then saves the transformed data into a
    new CSV file. The transformation involves checking for the existence of data for each
    lobster size and creating separate rows for them in the new dataframe.

    Args:
        input_csv_path (str): Path to the input CSV file containing the original analysis data.
        output_csv_path (str): Path where the transformed CSV file will be saved.

    Returns:
        None
    """
    # Load the analysis data from a CSV file
    df_analysis = pd.read_csv(input_csv_path)

    # Initialize a list to store new rows for the transformed dataframe
    new_rows = []

    # Iterate through each row in the original dataframe
    for _, row in df_analysis.iterrows():
        image_name = row['image_name']
        
        # Check and create a row for big lobster if data exists
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
        
        # Check and create a row for small lobster if data exists
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

    # Create a new dataframe from the list of new rows
    df_analysis = pd.DataFrame(new_rows)

    # Sort the new dataframe by image_name and lobster_size
    df_analysis = df_analysis.sort_values(['image_name', 'lobster_size']).reset_index(drop=True)

    # Save the sorted dataframe to a new CSV file
    df_analysis.to_csv(output_csv_path, index=False)

# Example usage
transform_and_save_analysis_data(
    'training and val output/runs/pose/predict83/length_analysis_new.csv',
    'training and val output/runs/pose/predict83/length_analysis_new_split.csv'
)
