import os
import shutil
from pathlib import Path

# Base directory
BASE_DIR = Path(os.getcwd())
TARGET_DIR = BASE_DIR / "data_files" / "consolidated_measurements"

# Ensure target directory exists
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# List of files to copy with their source paths
files_to_copy = [
    # Measurement data files
    "spreadsheet files/measurement/Filtered_Data.csv",
    "spreadsheet files/measurement/final_full_statistics_with_prawn_ids_and_uncertainty - Copy.xlsx",
    "spreadsheet files/measurement/test images.xlsx",
    
    # Exuviae analysis files
    "spreadsheet files/measurement/length_analysis_new_split.csv",
    "spreadsheet files/measurement/length_analysis_new_split_shai_exuviae.csv",
    "spreadsheet files/measurement/length_analysis_new_split_shai_exuviae_with_yolo.csv",
    "spreadsheet files/measurement/Results-shai-exuviae.csv",
]

# Files with variable types/weights
measurement_types = ['carapace', 'body']
weights_types = ['all', 'kalkar', 'car']

def copy_file(src_path, target_dir):
    """Copy a file to target directory if it exists"""
    src = BASE_DIR / src_path
    if src.exists():
        target = target_dir / src.name
        print(f"Copying {src.name} to {target_dir}")
        shutil.copy2(src, target)
    else:
        print(f"Warning: Source file not found: {src}")

# Copy fixed path files
for file_path in files_to_copy:
    copy_file(file_path, TARGET_DIR)

# Copy files with variable types
for mtype in measurement_types:
    # Copy error flags analysis files
    error_flags_file = f"spreadsheet files/measurement/error_flags_analysis_{mtype}_all_mean.csv"
    copy_file(error_flags_file, TARGET_DIR)
    
    # Copy filtered data files for each weights type
    for wtype in weights_types:
        filtered_data_file = f"spreadsheet files/measurement/updated_filtered_data_with_lengths_{mtype}-{wtype}.xlsx"
        copy_file(filtered_data_file, TARGET_DIR)

print("\nFile consolidation complete!") 