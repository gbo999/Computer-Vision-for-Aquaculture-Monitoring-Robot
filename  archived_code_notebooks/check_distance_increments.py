import pandas as pd
import numpy as np

def check_distance_increments():
    """
    Check distance values in 20mm increments from 600-800mm and calculate median ratio for Circle Pond.
    """
    
    print("=== Checking Distance Values in 20mm Increments ===")
    
    # Check the main exuviae data file
    try:
        df = pd.read_csv("spreadsheet_files/length_analysis_new_split_shai_exuviae.csv")
        print(f"File loaded: {len(df)} entries")
        print(f"Columns: {list(df.columns)}")
        
        # Check for distance and pond type columns
        if 'Pond_Type' in df.columns:
            print(f"\nPond types: {df['Pond_Type'].unique()}")
            
            # Filter for Circle pond
            circle_data = df[df['Pond_Type'] == 'Circle'].copy()
            print(f"Circle pond entries: {len(circle_data)}")
            
            # Look for distance-related columns
            distance_cols = [col for col in df.columns if 'distance' in col.lower() or 'Distance' in col or 'Height' in col]
            print(f"Distance columns found: {distance_cols}")
            
            for col in distance_cols:
                if col in circle_data.columns:
                    print(f"\n=== Circle Pond - {col} ===")
                    
                    # Check 20mm increments from 600-800
                    for start in range(600, 800, 20):
                        end = start + 20
                        mask = (circle_data[col] >= start) & (circle_data[col] < end)
                        
                        if mask.any():
                            subset = circle_data[mask]
                            print(f"\nRange {start}-{end}mm: {len(subset)} entries")
                            print(f"  Values: {subset[col].tolist()}")
                            print(f"  Median: {subset[col].median():.1f}")
                            
                            # Calculate ratio if we have error columns
                            if 'total_error_mm' in subset.columns and 'pixel_error_percent' in subset.columns:
                                ratio = subset['total_error_mm'] / subset['pixel_error_percent']
                                print(f"  Median ratio: {ratio.median():.3f}")
                                print(f"  Ratio range: {ratio.min():.3f} - {ratio.max():.3f}")
                    
                    # Overall statistics
                    print(f"\nOverall {col} statistics:")
                    print(f"  Range: {circle_data[col].min():.1f} - {circle_data[col].max():.1f}")
                    print(f"  Mean: {circle_data[col].mean():.1f}")
                    print(f"  Median: {circle_data[col].median():.1f}")
                    
                    # Calculate overall median ratio for Circle Pond
                    if 'total_error_mm' in circle_data.columns and 'pixel_error_percent' in circle_data.columns:
                        overall_ratio = circle_data['total_error_mm'] / circle_data['pixel_error_percent']
                        print(f"  Overall median ratio: {overall_ratio.median():.3f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_distance_increments() 