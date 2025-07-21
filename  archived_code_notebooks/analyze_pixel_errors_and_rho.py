import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_manual_length_pixels(rostrum_x, rostrum_y, tail_x, tail_y):
    img_width = 5312
    img_height = 2988
    
    rostrum_x_px = rostrum_x * img_width
    rostrum_y_px = rostrum_y * img_height
    tail_x_px = tail_x * img_width
    tail_y_px = tail_y * img_height
    
    length_px = np.sqrt((tail_x_px - rostrum_x_px)**2 + (tail_y_px - rostrum_y_px)**2)
    return length_px

# Load data
print("Loading merged data...")
merged_df = pd.read_csv("merged_manual_shai_keypoints_uniform_heights.csv")

# Filter valid data
valid_data = merged_df[
    (merged_df['shai_matched'] == True) & 
    (merged_df['keypoints_available'] == True) &
    (merged_df['calculated_total_length_mm'].notna()) &
    (merged_df['shai_length'].notna())
].copy()

# Calculate manual pixel lengths
print("\nCalculating manual pixel lengths...")
valid_data['manual_length_pixels'] = valid_data.apply(
    lambda row: calculate_manual_length_pixels(
        row['rostrum_x'], row['rostrum_y'],
        row['tail_x'], row['tail_y']
    ), axis=1
)

# Calculate pixel errors
print("\nCalculating pixel errors...")
valid_data['pixel_error'] = valid_data['shai_length'] - valid_data['manual_length_pixels']
valid_data['pixel_error_pct'] = (abs(valid_data['pixel_error']) / valid_data['manual_length_pixels']) * 100

# Identify outliers
print("\nIdentifying outliers (>30% pixel error)...")
valid_data['is_outlier'] = valid_data['pixel_error_pct'] > 30

# Calculate mm errors
print("\nCalculating mm errors...")
valid_data['mm_error'] = valid_data['calculated_total_length_mm'] - valid_data['reference_total_length']
valid_data['mm_error_pct'] = (abs(valid_data['mm_error']) / valid_data['reference_total_length']) * 100

# Print summary statistics
print("\n=== SUMMARY STATISTICS (UNIFORM HEIGHTS) ===")
print("Circle Pond: 690mm, Square Pond: 400mm")
print("-" * 100)
print(f"{'Pond':<8} {'Size':<7} {'Height':<8} {'N':<4} {'Med.Px.Err%':<12} {'Med.mm.Err%':<12} {'ρ':<8} {'Outliers'}")
print("-" * 100)

for pond_type in ['Circle', 'Square']:
    for size in ['small', 'big']:
        subset = valid_data[(valid_data['pond_type'] == pond_type) & 
                          (valid_data['manual_size'] == size)]
        
        if len(subset) > 0:
            # Calculate median percentage errors
            median_mm_error_pct = subset['mm_error_pct'].median()
            median_pixel_error_pct = subset['pixel_error_pct'].median()
            
            # Calculate ρ as ratio of medians
            rho = median_mm_error_pct / median_pixel_error_pct
            
            # Get height
            height = 690 if pond_type == 'Circle' else 400
            
            print(f"{pond_type:<8} {size:<7} {height:<8} {len(subset):<4} "
                  f"{median_pixel_error_pct:>6.1f}%      {median_mm_error_pct:>6.1f}%      "
                  f"{rho:>6.3f}  {subset['is_outlier'].sum()}({subset['is_outlier'].mean()*100:.1f}%)")

print("-" * 100)

# Save enhanced data
output_file = "enhanced_measurements_with_pixel_errors_uniform_heights.csv"
valid_data.to_csv(output_file, index=False)
print(f"\nEnhanced data saved to: {output_file}") 