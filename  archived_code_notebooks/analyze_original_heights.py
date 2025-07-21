import pandas as pd
import numpy as np

# Constants for height settings
CIRCLE_POND_HEIGHTS = {
    'small': 700,  # Original height for small exuviae in Circle pond
    'big': 660     # Original height for big exuviae in Circle pond
}

SQUARE_POND_HEIGHTS = {
    'small': 390,  # Original height for small exuviae in Square pond
    'big': 370     # Original height for big exuviae in Square pond
}

def get_pond_from_image(image_name):
    """Extract pond information from image name."""
    if 'GX010191' in image_name or 'GX010192' in image_name:
        return 'Circle'
    else:  # GX010193 or GX010194
        return 'Square'

def get_height_setting(row):
    """Get the appropriate height setting based on pond and size."""
    pond = get_pond_from_image(row['image_name'])
    if pond == 'Circle':
        return CIRCLE_POND_HEIGHTS['big'] if row['manual_size'] == 'big' else CIRCLE_POND_HEIGHTS['small']
    else:  # Square pond
        return SQUARE_POND_HEIGHTS['big'] if row['manual_size'] == 'big' else SQUARE_POND_HEIGHTS['small']

def calculate_mm_length(pixel_length, height, focal_length=2.92, sensor_height=4.8):
    """Calculate length in mm using the pinhole camera model."""
    return (pixel_length * height) / (focal_length * 1000)

def analyze_measurements():
    """Analyze measurements using original height settings."""
    # Load the merged data
    df = pd.read_csv('spreadsheet_files/merged_manual_shai_keypoints.csv')
    
    # Add pond information
    df['pond'] = df['image_name'].apply(get_pond_from_image)
    
    # Get appropriate height settings
    df['height'] = df.apply(get_height_setting, axis=1)
    
    # Calculate estimated length in mm using keypoint-based pixel length
    df['keypoint_pixel_length'] = np.sqrt(
        (df['tail_x'] - df['rostrum_x'])**2 + 
        (df['tail_y'] - df['rostrum_y'])**2
    ) * 1920  # Convert from normalized coordinates to pixels
    
    # Filter out rows where keypoints are missing
    df = df.dropna(subset=['keypoint_pixel_length', 'calculated_total_length_mm'])
    
    df['estimated_length_mm'] = df.apply(
        lambda row: calculate_mm_length(row['keypoint_pixel_length'], row['height']), 
        axis=1
    )
    
    # Calculate errors
    df['pixel_error'] = abs(df['keypoint_pixel_length'] - df['shai_length']) / df['shai_length'] * 100
    df['mm_error'] = abs(df['estimated_length_mm'] - df['calculated_total_length_mm']) / df['calculated_total_length_mm'] * 100
    
    # Group by pond and size
    grouped = df.groupby(['pond', 'manual_size'])
    
    # Calculate summary statistics
    summary_stats = grouped.agg({
        'pixel_error': ['median', 'mean', 'std', 'count'],
        'mm_error': ['median', 'mean', 'std']
    }).round(3)
    
    # Calculate ρ (rho) for each group
    summary_stats['rho'] = (summary_stats['mm_error']['mean'] / summary_stats['pixel_error']['mean']).round(3)
    
    # Print results
    print("\nAnalysis with original height settings:")
    print(f"Circle Pond - Small: {CIRCLE_POND_HEIGHTS['small']}mm, Big: {CIRCLE_POND_HEIGHTS['big']}mm")
    print(f"Square Pond - Small: {SQUARE_POND_HEIGHTS['small']}mm, Big: {SQUARE_POND_HEIGHTS['big']}mm\n")
    
    print("Summary Statistics:")
    print(summary_stats)
    
    # Calculate and print outlier percentages
    for (pond, size), group in df.groupby(['pond', 'manual_size']):
        outliers = len(group[group['mm_error'] > 10]) / len(group) * 100
        print(f"\n{pond} Pond ({size}):")
        print(f"Outliers (>10% error): {outliers:.1f}% ({len(group[group['mm_error'] > 10])} out of {len(group)})")
        print(f"Average ρ: {summary_stats.loc[(pond, size), 'rho']:.3f}")
        print(f"Median pixel error: {summary_stats.loc[(pond, size), ('pixel_error', 'median')]:.1f}%")
        print(f"Median mm error: {summary_stats.loc[(pond, size), ('mm_error', 'median')]:.1f}%")
    
    # Save detailed results
    df.to_csv('spreadsheet_files/analysis_results_original_heights.csv', index=False)

if __name__ == "__main__":
    analyze_measurements() 