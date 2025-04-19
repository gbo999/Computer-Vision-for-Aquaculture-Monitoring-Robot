import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the new data (name-first matching)
df_new = pd.read_excel('/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/combined_carapace_length_data.xlsx')

# Calculate median and MAD for the new data
cols = ['meas_scaled_meas_1', 'meas_scaled_meas_2', 'meas_scaled_meas_3', 
        'length_Length_1', 'length_Length_2', 'length_Length_3']
df_new['median_all'] = df_new[cols].median(axis=1)
df_new['MAD_all'] = df_new[cols].apply(lambda row: (row - row.median()).abs().median(), axis=1)

# Calculate errors for the new data
df_new['error'] = (df_new['length_Length_fov(mm)'] - df_new['median_all']).abs()
df_new['error_percent'] = (df_new['error'] / df_new['median_all']) * 100
df_new['error_to_MAD_ratio'] = df_new['error'] / df_new['MAD_all']

# Load the previous analysis results (if available)
try:
    # Create summary statistics for the new data
    new_stats = {
        'total_rows': len(df_new),
        'mean_error': df_new['error'].mean(),
        'median_error': df_new['error'].median(),
        'mean_percent_error': df_new['error_percent'].mean(),
        'mean_MAD': df_new['MAD_all'].mean(),
        'points_within_MAD': (df_new['error'] <= df_new['MAD_all']).mean() * 100,
        'points_within_2MAD': (df_new['error'] <= 2 * df_new['MAD_all']).mean() * 100,
        'points_within_5MAD': (df_new['error'] <= 5 * df_new['MAD_all']).mean() * 100,
        'match_types': df_new['match_type'].value_counts().to_dict(),
    }
    
    # Print summary comparison
    print("=== New Matching Method (Name First) ===")
    print(f"Total measurements: {new_stats['total_rows']}")
    print(f"Mean error: {new_stats['mean_error']:.2f} mm")
    print(f"Median error: {new_stats['median_error']:.2f} mm")
    print(f"Mean percentage error: {new_stats['mean_percent_error']:.2f}%")
    print(f"Mean MAD: {new_stats['mean_MAD']:.2f} mm")
    print(f"Points within 1x MAD: {new_stats['points_within_MAD']:.1f}%")
    print(f"Points within 2x MAD: {new_stats['points_within_2MAD']:.1f}%")
    print(f"Points within 5x MAD: {new_stats['points_within_5MAD']:.1f}%")
    
    print("\nMatch type distribution:")
    for match_type, count in new_stats['match_types'].items():
        print(f"  {match_type}: {count} ({count/new_stats['total_rows']*100:.1f}%)")
    
    # Create error distribution visualizations
    plt.figure(figsize=(12, 10))
    
    # Error distribution histogram
    plt.subplot(2, 2, 1)
    plt.hist(df_new['error'], bins=20, alpha=0.7)
    plt.title('Error Distribution (mm)')
    plt.xlabel('Absolute Error (mm)')
    plt.ylabel('Count')
    
    # Error-to-MAD ratio distribution
    plt.subplot(2, 2, 2)
    plt.hist(df_new['error_to_MAD_ratio'].clip(0, 20), bins=20, alpha=0.7)
    plt.title('Error-to-MAD Ratio Distribution (capped at 20)')
    plt.xlabel('Error / MAD Ratio')
    plt.ylabel('Count')
    
    # Scatter plot of measurements vs model
    plt.subplot(2, 2, 3)
    plt.scatter(df_new['median_all'], df_new['length_Length_fov(mm)'], alpha=0.5)
    
    # Add diagonal line
    min_val = min(df_new['median_all'].min(), df_new['length_Length_fov(mm)'].min())
    max_val = max(df_new['median_all'].max(), df_new['length_Length_fov(mm)'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add MAD bounds
    median_mad = df_new['MAD_all'].median()
    plt.plot([min_val, max_val], [min_val + median_mad, max_val + median_mad], 'g--', label=f'Median MAD: {median_mad:.2f}mm')
    plt.plot([min_val, max_val], [min_val - median_mad, max_val - median_mad], 'g--')
    
    plt.title('Model vs Median Measurements')
    plt.xlabel('Median of All Measurements (mm)')
    plt.ylabel('Model Measurement (mm)')
    plt.legend()
    plt.axis('equal')
    
    # Error vs MAD scatter
    plt.subplot(2, 2, 4)
    plt.scatter(df_new['MAD_all'], df_new['error'], alpha=0.5)
    
    # Add diagonal lines
    max_mad = df_new['MAD_all'].max()
    max_error = df_new['error'].max()
    upper_limit = max(max_mad, max_error) * 1.1
    
    plt.plot([0, upper_limit], [0, upper_limit], 'r--', label='Error = MAD')
    plt.plot([0, upper_limit], [0, 2*upper_limit], 'g--', label='Error = 2×MAD')
    plt.plot([0, upper_limit], [0, 5*upper_limit], 'b--', label='Error = 5×MAD')
    
    plt.title('Error vs MAD')
    plt.xlabel('MAD (mm)')
    plt.ylabel('Absolute Error (mm)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('name_first_matching_analysis.png')
    
    # Calculate percentage of error from outside MAD bounds
    outside_bounds = ~((df_new['length_Length_fov(mm)'] <= df_new['median_all'] + df_new['MAD_all']) & 
                      (df_new['length_Length_fov(mm)'] >= df_new['median_all'] - df_new['MAD_all']))
    
    outside_count = outside_bounds.sum()
    total_count = len(df_new)
    
    print(f"\nPoints outside MAD bounds: {outside_count} out of {total_count} ({outside_count/total_count*100:.1f}%)")
    
    if outside_count > 0:
        outside_errors = (df_new[outside_bounds]['length_Length_fov(mm)'] - 
                          df_new[outside_bounds]['median_all']).abs()
        
        total_outside_error = outside_errors.sum()
        total_error = df_new['error'].sum()
        
        print(f"Total error: {total_error:.2f} mm")
        print(f"Error from points outside MAD bounds: {total_outside_error:.2f} mm ({total_outside_error/total_error*100:.1f}% of total error)")
        print(f"Mean error for points outside MAD bounds: {outside_errors.mean():.2f} mm")
    
    plt.show()
    
except Exception as e:
    print(f"Error in analysis: {e}") 