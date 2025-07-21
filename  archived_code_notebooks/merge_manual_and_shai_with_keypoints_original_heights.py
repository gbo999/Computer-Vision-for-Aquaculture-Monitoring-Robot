import pandas as pd
import numpy as np
from pathlib import Path

# Constants for height settings
CIRCLE_POND_HEIGHTS = {
    'small': 700,  # Original height for small exuviae in Circle pond
    'big': 660     # Original height for big exuviae in Circle pond
}

SQUARE_POND_HEIGHTS = {
    'small': 390,  # Original height for small exuviae in Square pond
    'big': 370     # Original height for big exuviae in Square pond
}

def load_manual_measurements():
    """Load and preprocess manual measurements."""
    manual_df = pd.read_csv('spreadsheet_files/manual_classifications_with_bboxes.csv')
    # Convert columns to numeric, errors to NaN
    numeric_columns = ['length_mm', 'width_mm']  # Updated column names
    manual_df[numeric_columns] = manual_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Rename columns to match our expected schema
    manual_df = manual_df.rename(columns={
        'length_mm': 'manual_length_mm',
        'width_mm': 'manual_width_mm',
        'pixel_length': 'manual_pixel_length'
    })
    return manual_df

def load_automated_measurements():
    """Load and preprocess automated measurements."""
    automated_df = pd.read_csv('spreadsheet_files/shai_measurements_with_keypoints.csv')
    return automated_df

def calculate_pixel_length(row):
    """Calculate pixel length using keypoints."""
    x1, y1 = row['kp_head_x'], row['kp_head_y']
    x2, y2 = row['kp_telson_x'], row['kp_telson_y']
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_height_setting(row):
    """Get the appropriate height setting based on pond and size."""
    if row['pond'] == 'Circle':
        return CIRCLE_POND_HEIGHTS['big'] if row['size'] == 'big' else CIRCLE_POND_HEIGHTS['small']
    else:  # Square pond
        return SQUARE_POND_HEIGHTS['big'] if row['size'] == 'big' else SQUARE_POND_HEIGHTS['small']

def calculate_mm_length(pixel_length, height, focal_length=2.92, sensor_height=4.8):
    """Calculate length in mm using the pinhole camera model."""
    return (pixel_length * height) / (focal_length * 1000)

def merge_and_analyze():
    """Merge manual and automated measurements and perform analysis."""
    manual_df = load_manual_measurements()
    automated_df = load_automated_measurements()
    
    # Merge the dataframes
    merged_df = pd.merge(manual_df, automated_df, on='image_id', suffixes=('_manual', '_auto'))
    
    # Calculate pixel length
    merged_df['pixel_length'] = merged_df.apply(calculate_pixel_length, axis=1)
    
    # Get appropriate height settings
    merged_df['height'] = merged_df.apply(get_height_setting, axis=1)
    
    # Calculate estimated length in mm
    merged_df['estimated_length_mm'] = merged_df.apply(
        lambda row: calculate_mm_length(row['pixel_length'], row['height']), 
        axis=1
    )
    
    # Calculate errors
    merged_df['pixel_error'] = abs(merged_df['pixel_length'] - merged_df['manual_pixel_length']) / merged_df['manual_pixel_length'] * 100
    merged_df['mm_error'] = abs(merged_df['estimated_length_mm'] - merged_df['manual_length_mm']) / merged_df['manual_length_mm'] * 100
    
    # Group by pond and size
    grouped = merged_df.groupby(['pond', 'size'])
    
    # Calculate summary statistics
    summary_stats = grouped.agg({
        'pixel_error': ['median', 'mean', 'std', 'count'],
        'mm_error': ['median', 'mean', 'std']
    }).round(3)
    
    # Calculate ρ (rho) for each group
    summary_stats['rho'] = (summary_stats['mm_error']['mean'] / summary_stats['pixel_error']['mean']).round(3)
    
    # Save results
    merged_df.to_csv('spreadsheet_files/merged_measurements_original_heights.csv', index=False)
    summary_stats.to_csv('spreadsheet_files/summary_stats_original_heights.csv')
    
    return merged_df, summary_stats

def main():
    """Main function to run the analysis."""
    print("Running analysis with original height settings:")
    print(f"Circle Pond - Small: {CIRCLE_POND_HEIGHTS['small']}mm, Big: {CIRCLE_POND_HEIGHTS['big']}mm")
    print(f"Square Pond - Small: {SQUARE_POND_HEIGHTS['small']}mm, Big: {SQUARE_POND_HEIGHTS['big']}mm\n")
    
    _, summary_stats = merge_and_analyze()
    
    print("\nSummary Statistics:")
    print(summary_stats)
    
    # Calculate and print outlier percentages
    merged_df, _ = merge_and_analyze()
    for (pond, size), group in merged_df.groupby(['pond', 'size']):
        outliers = len(group[group['mm_error'] > 10]) / len(group) * 100
        print(f"\n{pond} Pond ({size}):")
        print(f"Outliers (>10% error): {outliers:.1f}% ({len(group[group['mm_error'] > 10])} out of {len(group)})")

if __name__ == "__main__":
    main() 