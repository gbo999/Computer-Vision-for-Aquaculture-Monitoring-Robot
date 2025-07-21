import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_manual_length_pixels(rostrum_x, rostrum_y, tail_x, tail_y):
    """Calculate manual length in pixels from keypoint coordinates."""
    img_width = 5312
    img_height = 2988
    
    rostrum_x_px = rostrum_x * img_width
    rostrum_y_px = rostrum_y * img_height
    tail_x_px = tail_x * img_width
    tail_y_px = tail_y * img_height
    
    length_px = np.sqrt((tail_x_px - rostrum_x_px)**2 + (tail_y_px - rostrum_y_px)**2)
    return length_px

def create_comprehensive_table(valid_data, clean_data):
    """Create comprehensive table with MAE, MAPE, and pixel error only - NO RHO."""
    print("\n" + "="*100)
    print("COMPREHENSIVE ANALYSIS TABLE - 700mm HEIGHT SETTING (NO RHO)")
    print("="*100)
    
    # Initialize results dictionary
    results = {}
    
    for pond_type in ['Circle', 'Square']:
        for size in ['small', 'big']:
            subset = valid_data[(valid_data['pond_type'] == pond_type) & 
                              (valid_data['manual_size'] == size)]
            subset_clean = clean_data[(clean_data['pond_type'] == pond_type) & 
                                    (clean_data['manual_size'] == size)]
            
            if len(subset) > 0:
                # Calculate metrics
                n_samples = len(subset)
                n_clean = len(subset_clean)
                
                # Outliers (>30% pixel error)
                outliers_count = subset['is_outlier'].sum()
                outliers_pct = (outliers_count / n_samples) * 100
                
                # Problematic measurements
                problematic_count = subset['is_problematic'].sum()
                problematic_pct = (problematic_count / n_samples) * 100
                
                # Use clean data for calculations (no outliers, no problematic)
                if len(subset_clean) > 0:
                    median_mae_clean = abs(subset_clean['mm_error']).median()
                    median_pixel_error_clean = subset_clean['pixel_error_pct'].median()
                    median_mape_clean = subset_clean['mm_error_pct'].median()
                else:
                    median_mae_clean = float('nan')
                    median_pixel_error_clean = float('nan')
                    median_mape_clean = float('nan')
                
                # Keep original calculations for comparison
                median_mae_without_outliers = abs(subset[~subset['is_outlier']]['mm_error']).median()
                median_mae_with_outliers = abs(subset['mm_error']).median()
                median_pixel_error_without_outliers = subset[~subset['is_outlier']]['pixel_error_pct'].median()
                median_pixel_error_with_outliers = subset['pixel_error_pct'].median()
                median_mape_without_outliers = subset[~subset['is_outlier']]['mm_error_pct'].median()
                median_mape_with_outliers = subset['mm_error_pct'].median()
                
                print(f"DEBUG: {pond_type} {size} median_mape_clean = {median_mape_clean}")
                results[f"{pond_type}_{size}"] = {
                    'Pond': pond_type,
                    'Size': size,
                    'N': n_samples,
                    'Clean_N': n_clean,
                    'Outliers': f"{outliers_count} ({outliers_pct:.1f}%)",
                    'Problematic': f"{problematic_count} ({problematic_pct:.1f}%)",
                    'Median MAE (mm)': f"{median_mae_with_outliers:.2f}",
                    'Median MAE no outliers (mm)': f"{median_mae_without_outliers:.2f}",
                    'Median MAE clean (mm)': f"{median_mae_clean:.2f}",
                    'Median MAPE (%)': f"{median_mape_with_outliers:.2f}",
                    'Median MAPE no outliers (%)': f"{median_mape_without_outliers:.2f}",
                    'Median MAPE clean (%)': f"{median_mape_clean:.2f}",
                    'Median Pixel Error (%)': f"{median_pixel_error_with_outliers:.1f}",
                    'Median Pixel Error no outliers (%)': f"{median_pixel_error_without_outliers:.1f}",
                    'Median Pixel Error clean (%)': f"{median_pixel_error_clean:.1f}"
                }
    
    # Create DataFrame and display table
    results_df = pd.DataFrame(results).T
    
    # Reorder columns
    column_order = ['Pond', 'Size', 'N', 'Clean_N', 'Outliers', 'Problematic', 'Median MAE (mm)', 'Median MAE no outliers (mm)', 'Median MAE clean (mm)', 'Median MAPE (%)', 'Median MAPE no outliers (%)', 'Median MAPE clean (%)', 'Median Pixel Error (%)', 'Median Pixel Error no outliers (%)', 'Median Pixel Error clean (%)']
    results_df = results_df[column_order]
    
    print(results_df.to_string(index=False, max_colwidth=15))
    
    # Also print a simplified version with key metrics
    print("\n" + "="*100)
    print("SIMPLIFIED TABLE - KEY METRICS (CLEAN DATA)")
    print("="*100)
    simple_columns = ['Pond', 'Size', 'N', 'Clean_N', 'Median MAE clean (mm)', 'Median MAPE clean (%)', 'Median Pixel Error clean (%)']
    print(results_df[simple_columns].to_string(index=False))
    
    # Print summary by pond type
    print("\n" + "="*100)
    print("SUMMARY BY POND TYPE (CLEAN DATA)")
    print("="*100)
    
    for pond_type in ['Circle', 'Square']:
        pond_data_clean = clean_data[clean_data['pond_type'] == pond_type]
        if len(pond_data_clean) > 0:
            print(f"\n{pond_type} POND SUMMARY (CLEAN):")
            print(f"Clean samples: {len(pond_data_clean)}")
            print(f"Overall Median MAE: {abs(pond_data_clean['mm_error']).median():.2f} mm")
            print(f"Overall Median MAPE: {pond_data_clean['mm_error_pct'].median():.2f}%")
            print(f"Overall Median pixel error: {pond_data_clean['pixel_error_pct'].median():.1f}%")
    
    return results_df

def mad(series):
    """Median absolute deviation."""
    return np.median(np.abs(series - np.median(series)))

def create_small_exuviae_table(valid_data):
    """Create a table for small exuviae only, using only outlier filtering (pixel error > 30%), reporting median ± MAD."""
    print("\n" + "="*100)
    print("SMALL EXUVIAE ANALYSIS TABLE - 700mm HEIGHT SETTING (OUTLIERS ONLY, MEDIAN ± MAD)\n" + "="*100)
    
    results = {}
    for pond_type in ['Circle', 'Square']:
        subset = valid_data[(valid_data['pond_type'] == pond_type) & (valid_data['manual_size'] == 'small')]
        n_total = len(subset)
        n_outliers = subset['is_outlier'].sum()
        n_no_outliers = n_total - n_outliers
        subset_no_outliers = subset[~subset['is_outlier']]
        
        if n_no_outliers > 0:
            mae = abs(subset_no_outliers['mm_error'])
            mape = subset_no_outliers['mm_error_pct']
            pixel_error = subset_no_outliers['pixel_error_pct']
            
            results[pond_type] = {
                'Pond': pond_type,
                'N_total': n_total,
                'N_no_outliers': n_no_outliers,
                'Outliers': n_outliers,
                'MAE (mm)': f"{mae.median():.2f} ± {mad(mae):.2f}",
                'MAPE (%)': f"{mape.median():.2f} ± {mad(mape):.2f}",
                'Pixel error (%)': f"{pixel_error.median():.2f} ± {mad(pixel_error):.2f}"
            }
    
    import pandas as pd
    results_df = pd.DataFrame(results).T
    print(results_df.to_string(index=False, max_colwidth=15))
    return results_df

def analyze_measurements():
    """Analyze measurements using the exact method from previous analysis."""
    print("Loading merged data...")
    df = pd.read_csv('spreadsheet_files/merged_manual_shai_keypoints.csv')
    
    # Filter valid data (same as previous method)
    valid_data = df[
        (df['shai_matched'] == True) & 
        (df['keypoints_available'] == True) &
        (df['calculated_total_length_mm'].notna()) &
        (df['shai_length'].notna())
    ].copy()
    
    # Add reference lengths based on size (since they're missing from original file)
    valid_data['reference_total_length'] = valid_data['manual_size'].map({'small': 145, 'big': 180})
    valid_data['reference_carapace_length'] = valid_data['manual_size'].map({'small': 41, 'big': 50})
    
    # Add pond_type column based on image_name
    valid_data['pond_type'] = valid_data['image_name'].apply(lambda x: 'Circle' if 'GX010191' in x else 'Square')
    
    print(f"Valid measurements: {len(valid_data)}")
    
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
    
    # Identify outliers (>30% pixel error)
    print("\nIdentifying outliers (>30% pixel error)...")
    valid_data['is_outlier'] = valid_data['pixel_error_pct'] > 30
    
    # ADDITIONAL FILTERING: Remove measurements with unrealistic pixel errors
    print("\nIdentifying problematic measurements (unrealistic pixel errors)...")
    
    # Mark measurements with unrealistic pixel errors as problematic
    valid_data['is_problematic'] = (
        (valid_data['pixel_error_pct'] < 0.1)  # pixel error < 0.1% is suspiciously small
    )
    
    # Show problematic measurements
    problematic = valid_data[valid_data['is_problematic']]
    if len(problematic) > 0:
        print(f"Found {len(problematic)} problematic measurements:")
        for idx, row in problematic.iterrows():
            print(f"  {row['image_name']}: pixel_error_pct = {row['pixel_error_pct']:.3f}% (suspiciously small)")
    
    # Calculate mm errors
    print("\nCalculating mm errors...")
    valid_data['mm_error'] = valid_data['calculated_total_length_mm'] - valid_data['reference_total_length']
    valid_data['mm_error_pct'] = (abs(valid_data['mm_error']) / valid_data['reference_total_length']) * 100
    
    # Create clean dataset excluding both outliers and problematic measurements
    valid_data['is_clean'] = ~(valid_data['is_outlier'] | valid_data['is_problematic'])
    clean_data = valid_data[valid_data['is_clean']].copy()
    
    print(f"\nClean measurements: {len(clean_data)} out of {len(valid_data)} total")
    print(f"Excluded: {len(valid_data) - len(clean_data)} measurements (outliers + problematic)")
    
    # Create comprehensive table using clean data
    results_table = create_comprehensive_table(valid_data, clean_data)
    
    # Print summary statistics (same format as previous)
    print("\n=== SUMMARY STATISTICS (700mm HEIGHT) ===")
    print("Circle Pond: 700mm, Square Pond: 390mm")
    print("-" * 100)
    print(f"{'Pond':<8} {'Size':<7} {'Height':<8} {'N':<4} {'Med.Px.Err%':<12} {'Med.mm.Err%':<12} {'ρ':<8} {'Outliers'}")
    print("-" * 100)
    
    for pond_type in ['Circle', 'Square']:
        for size in ['small', 'big']:
            subset = valid_data[(valid_data['pond_type'] == pond_type) & 
                              (valid_data['manual_size'] == size)]
            
            if len(subset) > 0:
                # Calculate median percentage errors (same as previous method)
                median_mm_error_pct = subset['mm_error_pct'].median()
                median_pixel_error_pct = subset['pixel_error_pct'].median()
                
                # Calculate ρ as ratio of medians (same as previous method)
                rho = median_mm_error_pct / median_pixel_error_pct
                
                # Get height
                height = 700 if pond_type == 'Circle' else 390
                
                print(f"{pond_type:<8} {size:<7} {height:<8} {len(subset):<4} "
                      f"{median_pixel_error_pct:>6.1f}%      {median_mm_error_pct:>6.1f}%      "
                      f"{rho:>6.3f}  {subset['is_outlier'].sum()}({subset['is_outlier'].mean()*100:.1f}%)")
    
    print("-" * 100)
    
    # Save enhanced data
    output_file = "enhanced_measurements_with_pixel_errors_700mm.csv"
    valid_data.to_csv(output_file, index=False)
    print(f"\nEnhanced data saved to: {output_file}")
    
    # Save results table
    table_file = "comprehensive_analysis_table_700mm.csv"
    results_table.to_csv(table_file, index=False)
    print(f"Comprehensive table saved to: {table_file}")
    
    # Print detailed analysis for small exuviae only (as requested)
    print("\n=== DETAILED ANALYSIS - SMALL EXUVIAE ONLY ===")
    print("Focusing on Circle small and Square small exuviae:")
    
    for pond_type in ['Circle', 'Square']:
        subset_clean = clean_data[(clean_data['pond_type'] == pond_type) & 
                                (clean_data['manual_size'] == 'small')]
        
        if len(subset_clean) > 0:
            print(f"\n{pond_type} Pond - Small Exuviae (CLEAN):")
            print(f"Clean sample size: {len(subset_clean)}")
            print(f"Median pixel error: {subset_clean['pixel_error_pct'].median():.1f}%")
            print(f"Median MAPE: {subset_clean['mm_error_pct'].median():.1f}%")
            print(f"Median MAE: {abs(subset_clean['mm_error']).median():.2f} mm")
    
    # Save clean data for further analysis
    clean_data[['image_name', 'pond_type', 'manual_size', 'mm_error_pct', 'pixel_error_pct']].to_csv('clean_measurements_700mm.csv', index=False)
    print(f"\nClean measurements saved to: clean_measurements_700mm.csv")

    # Only outlier filtering
    small_exuviae_table = create_small_exuviae_table(valid_data)
    
    # Write summary paragraph
    for pond_type in ['Circle', 'Square']:
        row = small_exuviae_table.loc[pond_type]
        print(f"\n---\n{pond_type} Pond - Small Exuviae:")
        print(f"Total measurements: {row['N_total']}")
        print(f"Outliers removed (pixel error > 30%): {row['Outliers']}")
        print(f"Measurements used for analysis: {row['N_no_outliers']}")
        print(f"MAE (median ± MAD): {row['MAE (mm)']} mm")
        print(f"MAPE (median ± MAD): {row['MAPE (%)']}%")
        print(f"Pixel error (median ± MAD): {row['Pixel error (%)']}%")
    
    print("\nParagraph for methods/results:")
    print("For each pond, we analyzed only small exuviae. Outliers were defined as measurements with pixel error greater than 30% and were excluded from the analysis. For Circle pond, {0} out of {1} measurements were removed as outliers; for Square pond, {2} out of {3} measurements were removed. All reported statistics (median ± MAD for MAE, MAPE, and pixel error) are based on the remaining measurements after outlier removal.".format(
        int(small_exuviae_table.loc['Circle']['Outliers']),
        int(small_exuviae_table.loc['Circle']['N_total']),
        int(small_exuviae_table.loc['Square']['Outliers']),
        int(small_exuviae_table.loc['Square']['N_total'])
    ))

if __name__ == "__main__":
    analyze_measurements() 