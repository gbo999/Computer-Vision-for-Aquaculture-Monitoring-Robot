import pandas as pd
import numpy as np
import os
from pathlib import Path

# Create output directory if it doesn't exist
output_dir = Path('runs/pose/predict57')
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data
df = pd.read_csv('runs/pose/predict57/length_analysis.csv')

# Add pond type column
df['pond_type'] = df['image_name'].apply(lambda x: 'Circle' if '10191' in x else 'Square')

# Expected values
expected_big_total = 180  # mm
expected_small_total = 145  # mm
expected_big_carapace = 63  # mm
expected_small_carapace = 41  # mm

# Function to calculate errors for a specific pond type
def calculate_pond_errors(pond_df, pond_type):
    # Calculate absolute errors
    big_total_abs_errors = abs(pond_df['big_total_length'] - expected_big_total)
    small_total_abs_errors = abs(pond_df['small_total_length'] - expected_small_total)
    big_carapace_abs_errors = abs(pond_df['big_carapace_length'] - expected_big_carapace)
    small_carapace_abs_errors = abs(pond_df['small_carapace_length'] - expected_small_carapace)

    # Calculate percentage errors
    big_total_perc_errors = (big_total_abs_errors / expected_big_total) * 100
    small_total_perc_errors = (small_total_abs_errors / expected_small_total) * 100
    big_carapace_perc_errors = (big_carapace_abs_errors / expected_big_carapace) * 100
    small_carapace_perc_errors = (small_carapace_abs_errors / expected_small_carapace) * 100

    return pd.DataFrame({
        'Pond Type': [pond_type, pond_type],
        'Prawn Type': ['Big', 'Small'],
        'Ground Truth Total Length (mm)': [expected_big_total, expected_small_total],
        'MAE Total Length (mm)': [
            big_total_abs_errors.median(),
            small_total_abs_errors.median()
        ],
        'MAPE Total Length (%)': [
            big_total_perc_errors.median(),
            small_total_perc_errors.median()
        ],
        'Ground Truth Carapace Length (mm)': [expected_big_carapace, expected_small_carapace],
        'MAE Carapace Length (mm)': [
            big_carapace_abs_errors.median(),
            small_carapace_abs_errors.median()
        ],
        'MAPE Carapace Length (%)': [
            big_carapace_perc_errors.median(),
            small_carapace_perc_errors.median()
        ],
        'Total Length Count': [
            pond_df['big_total_length'].notna().sum(),
            pond_df['small_total_length'].notna().sum()
        ],
        'Carapace Length Count': [
            pond_df['big_carapace_length'].notna().sum(),
            pond_df['small_carapace_length'].notna().sum()
        ]
    })

# Calculate errors for each pond type
circle_df = calculate_pond_errors(df[df['pond_type'] == 'Circle'], 'Circle')
square_df = calculate_pond_errors(df[df['pond_type'] == 'Square'], 'Square')

# Combine results
results_df = pd.concat([circle_df, square_df], ignore_index=True)

# Format float columns to 2 decimal places
float_columns = results_df.select_dtypes(include=['float64']).columns
results_df[float_columns] = results_df[float_columns].round(2)

# Save to CSV with full path
output_file = output_dir / 'mae_mape_results.csv'
results_df.to_csv(output_file, index=False)

# Display the results and file location
print("\nResults DataFrame:")
print(results_df.to_string(index=False))
print(f"\nResults have been saved to: {output_file.absolute()}")

# Verify file was created
if output_file.exists():
    print("File was successfully created!")
else:
    print("Error: File was not created!")