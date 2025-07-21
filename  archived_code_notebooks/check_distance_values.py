import pandas as pd
import numpy as np

def check_distance_values():
    """
    Check distance values between 600-800mm and calculate median ratio for Circle Pond.
    """
    
    # Check what files have distance information
    print("=== Checking Distance Values ===")
    
    # Look for files with pred_Distance_pixels
    files_to_check = [
        "spreadsheet_files/merged_manual_shai_keypoints.csv",
        "spreadsheet_files/Results-shai-exuviae.csv",
        "spreadsheet_files/length_analysis_new_split_shai_exuviae.csv"
    ]
    
    for file_path in files_to_check:
        try:
            df = pd.read_csv(file_path)
            print(f"\nFile: {file_path}")
            print(f"Columns: {list(df.columns)}")
            
            # Check for distance-related columns
            distance_cols = [col for col in df.columns if 'distance' in col.lower() or 'Distance' in col]
            if distance_cols:
                print(f"Distance columns found: {distance_cols}")
                
                for col in distance_cols:
                    if col in df.columns:
                        print(f"\n{col} statistics:")
                        print(f"  Range: {df[col].min():.2f} - {df[col].max():.2f}")
                        print(f"  Mean: {df[col].mean():.2f}")
                        print(f"  Median: {df[col].median():.2f}")
                        
                        # Check values between 600-800
                        mask_600_800 = (df[col] >= 600) & (df[col] <= 800)
                        if mask_600_800.any():
                            values_600_800 = df[col][mask_600_800]
                            print(f"  Values 600-800: {len(values_600_800)} entries")
                            print(f"  Range 600-800: {values_600_800.min():.2f} - {values_600_800.max():.2f}")
                            print(f"  Mean 600-800: {values_600_800.mean():.2f}")
                            print(f"  Median 600-800: {values_600_800.median():.2f}")
                        else:
                            print(f"  No values between 600-800")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Check if we have Circle Pond data with distance
    print("\n=== Looking for Circle Pond Distance Data ===")
    
    # Try to find files with Pond_Type information
    pond_files = [
        "spreadsheet_files/length_analysis_new_split_shai_exuviae.csv",
        "spreadsheet_files/merged_manual_shai_keypoints.csv"
    ]
    
    for file_path in pond_files:
        try:
            df = pd.read_csv(file_path)
            if 'Pond_Type' in df.columns:
                print(f"\nFile: {file_path}")
                print(f"Pond types: {df['Pond_Type'].unique()}")
                
                # Check for Circle pond
                if 'Circle' in df['Pond_Type'].values:
                    circle_data = df[df['Pond_Type'] == 'Circle']
                    print(f"Circle pond entries: {len(circle_data)}")
                    
                    # Look for distance columns
                    distance_cols = [col for col in df.columns if 'distance' in col.lower() or 'Distance' in col]
                    for col in distance_cols:
                        if col in circle_data.columns:
                            print(f"\nCircle Pond - {col}:")
                            print(f"  Range: {circle_data[col].min():.2f} - {circle_data[col].max():.2f}")
                            print(f"  Mean: {circle_data[col].mean():.2f}")
                            print(f"  Median: {circle_data[col].median():.2f}")
                            
                            # Check 600-800 range
                            mask_600_800 = (circle_data[col] >= 600) & (circle_data[col] <= 800)
                            if mask_600_800.any():
                                values_600_800 = circle_data[col][mask_600_800]
                                print(f"  Values 600-800: {len(values_600_800)} entries")
                                print(f"  Median 600-800: {values_600_800.median():.2f}")
                                
                                # Calculate ratio if we have another relevant column
                                if 'total_error_mm' in circle_data.columns and 'pixel_error_percent' in circle_data.columns:
                                    ratio = circle_data['total_error_mm'] / circle_data['pixel_error_percent']
                                    print(f"  Median ratio (total_error/pixel_error): {ratio.median():.3f}")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    check_distance_values() 