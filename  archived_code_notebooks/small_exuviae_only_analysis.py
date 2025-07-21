import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def small_exuviae_only_analysis():
    """
    Analysis of Small Exuviae Measurements Only
    
    IMPORTANT CONTEXT:
    - We have only 2 exuviae objects total: 1 big, 1 small
    - The big exuviae is excluded due to poor quality/morphology (doesn't look like a prawn)
    - The small exuviae is good quality and suitable for measurement
    - Multiple measurements exist for each exuviae from different methods and images
    
    This analysis focuses exclusively on the small exuviae measurements to evaluate
    measurement accuracy and reliability.
    """
    
    print("=" * 80)
    print("SMALL EXUVIAE MEASUREMENT ANALYSIS")
    print("=" * 80)
    print("📊 Analysis focusing on small exuviae only")
    print("🔬 Context: Only 2 exuviae total (1 big, 1 small)")
    print("❌ Big exuviae excluded due to poor quality/morphology")
    print("✅ Small exuviae included due to good quality")
    print("=" * 80)
    
    # Load the merged dataset
    df = pd.read_csv("spreadsheet_files/merged_manual_shai_keypoints.csv")
    print(f"✅ Loaded {len(df)} total measurements from merged CSV")
    
    # Filter for small prawns only
    small_df = df[df['manual_size'] == 'small'].copy()
    print(f"📏 Filtered to {len(small_df)} small exuviae measurements")
    
    # Add pond type based on image name
    small_df['pond_type'] = small_df['image_name'].apply(lambda x: 'Circle' if 'GX010191' in x else 'Square')
    
    # Set reference length for small prawns
    SMALL_REF_TOTAL = 145  # mm
    SMALL_REF_CARAPACE = 41  # mm
    small_df['reference_total_length'] = SMALL_REF_TOTAL
    small_df['reference_carapace_length'] = SMALL_REF_CARAPACE
    
    # Calculate total length in pixels from keypoints (rostrum to tail distance)
    img_width_px = 5312
    img_height_px = 2988
    
    # Convert normalized coordinates to pixels and calculate distance
    small_df['rostrum_x_px'] = small_df['rostrum_x'] * img_width_px
    small_df['rostrum_y_px'] = small_df['rostrum_y'] * img_height_px
    small_df['tail_x_px'] = small_df['tail_x'] * img_width_px
    small_df['tail_y_px'] = small_df['tail_y'] * img_height_px
    
    small_df['total_length_pixels'] = np.sqrt((small_df['rostrum_x_px'] - small_df['tail_x_px'])**2 + 
                                             (small_df['rostrum_y_px'] - small_df['tail_y_px'])**2)
    
    # Calculate errors
    small_df['total_error_mm'] = small_df['calculated_total_length_mm'] - SMALL_REF_TOTAL
    small_df['carapace_error_mm'] = small_df['calculated_carapace_length_mm'] - SMALL_REF_CARAPACE
    small_df['total_error_pct'] = (small_df['total_error_mm'] / SMALL_REF_TOTAL) * 100
    small_df['carapace_error_pct'] = (small_df['carapace_error_mm'] / SMALL_REF_CARAPACE) * 100
    
    # Calculate pixel error (difference between manual and Shai pixel measurements)
    small_df['pixel_error'] = abs(small_df['total_length_pixels'] - small_df['shai_length'])
    small_df['pixel_error_pct'] = (small_df['pixel_error'] / small_df['total_length_pixels']) * 100
    
    # Calculate scale error (how scaling affects measurement)
    small_df['scale_error'] = abs(small_df['total_error_pct'] - small_df['pixel_error_pct'])
    
    # Remove outliers using IQR method
    def remove_outliers_iqr(data, column, factor=1.5):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Remove outliers from key error metrics
    small_df_clean = small_df.copy()
    for col in ['total_error_mm', 'carapace_error_mm', 'pixel_error']:
        if col in small_df_clean.columns:
            small_df_clean = remove_outliers_iqr(small_df_clean, col)
    
    print(f"🧹 After outlier removal: {len(small_df_clean)} measurements")
    print(f"🗑️ Removed {len(small_df) - len(small_df_clean)} outlier measurements")
    
    # Create comprehensive measurement table
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE MEASUREMENT TABLE - SMALL EXUVIAE ONLY")
    print("=" * 80)
    
    # Group by pond type and calculate statistics
    pond_stats = []
    
    for pond_type in ['Circle', 'Square']:
        pond_data = small_df_clean[small_df_clean['pond_type'] == pond_type]
        
        if len(pond_data) == 0:
            continue
            
        stats_row = {
            'Pond Type': pond_type,
            'Count': len(pond_data),
            'MAE Total (mm)': np.median(np.abs(pond_data['total_error_mm'])),
            'MAE Carapace (mm)': np.median(np.abs(pond_data['carapace_error_mm'])),
            'MAPE Total (%)': np.median(np.abs(pond_data['total_error_pct'])),
            'MAPE Carapace (%)': np.median(np.abs(pond_data['carapace_error_pct'])),
            'Pixel Error (px)': np.median(pond_data['pixel_error']),
            'Scale Error (%)': np.median(pond_data['scale_error'])
        }
        pond_stats.append(stats_row)
    
    # Overall statistics
    overall_stats = {
        'Pond Type': 'Overall',
        'Count': len(small_df_clean),
        'MAE Total (mm)': np.median(np.abs(small_df_clean['total_error_mm'])),
        'MAE Carapace (mm)': np.median(np.abs(small_df_clean['carapace_error_mm'])),
        'MAPE Total (%)': np.median(np.abs(small_df_clean['total_error_pct'])),
        'MAPE Carapace (%)': np.median(np.abs(small_df_clean['carapace_error_pct'])),
        'Pixel Error (px)': np.median(small_df_clean['pixel_error']),
        'Scale Error (%)': np.median(small_df_clean['scale_error'])
    }
    pond_stats.append(overall_stats)
    
    # Create and display table
    stats_df = pd.DataFrame(pond_stats)
    print(stats_df.to_string(index=False, float_format='%.2f'))
    
    # Save detailed results
    print("\n" + "=" * 80)
    print("📋 DETAILED MEASUREMENT RESULTS")
    print("=" * 80)
    
    # Create detailed results table
    detailed_results = small_df_clean[['image_name', 'pond_type', 'calculated_total_length_mm', 
                                      'calculated_carapace_length_mm', 'total_error_mm', 'carapace_error_mm',
                                      'total_error_pct', 'carapace_error_pct', 'pixel_error', 'scale_error']].copy()
    
    detailed_results = detailed_results.round(2)
    detailed_results.columns = ['Image', 'Pond', 'Pred_Total_mm', 'Pred_Carapace_mm', 
                               'Total_Error_mm', 'Carapace_Error_mm', 'Total_Error_%', 
                               'Carapace_Error_%', 'Pixel_Error_px', 'Scale_Error_%']
    
    print(detailed_results.to_string(index=False))
    
    # Save to file
    detailed_results.to_csv('small_exuviae_detailed_results.csv', index=False)
    stats_df.to_csv('small_exuviae_summary_table.csv', index=False)
    
    print(f"\n💾 Results saved to:")
    print(f"   - small_exuviae_detailed_results.csv")
    print(f"   - small_exuviae_summary_table.csv")
    
    # Statistical analysis
    print("\n" + "=" * 80)
    print("🔬 STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Test for differences between pond types
    circle_data = small_df_clean[small_df_clean['pond_type'] == 'Circle']
    square_data = small_df_clean[small_df_clean['pond_type'] == 'Square']
    
    if len(circle_data) > 0 and len(square_data) > 0:
        # Mann-Whitney U test for total error
        stat, p_value = stats.mannwhitneyu(circle_data['total_error_mm'], square_data['total_error_mm'], alternative='two-sided')
        print(f"📊 Mann-Whitney U test (Total Error): p = {p_value:.4f}")
        print(f"   {'Significant difference' if p_value < 0.05 else 'No significant difference'} between pond types")
        
        # Mann-Whitney U test for carapace error
        stat, p_value = stats.mannwhitneyu(circle_data['carapace_error_mm'], square_data['carapace_error_mm'], alternative='two-sided')
        print(f"📊 Mann-Whitney U test (Carapace Error): p = {p_value:.4f}")
        print(f"   {'Significant difference' if p_value < 0.05 else 'No significant difference'} between pond types")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   Total measurements analyzed: {len(small_df_clean)}")
    print(f"   Circle pond measurements: {len(circle_data)}")
    print(f"   Square pond measurements: {len(square_data)}")
    print(f"   Reference total length: {SMALL_REF_TOTAL} mm")
    print(f"   Reference carapace length: {SMALL_REF_CARAPACE} mm")
    
    print("\n" + "=" * 80)
    print("🎯 KEY FINDINGS")
    print("=" * 80)
    print("1. Analysis focuses on small exuviae only (good quality)")
    print("2. Big exuviae excluded due to poor morphology/quality")
    print("3. Outliers removed using IQR method for robust statistics")
    print("4. Median-based analysis for reliable conclusions")
    print("5. Comprehensive error metrics calculated")
    print("=" * 80)

if __name__ == "__main__":
    small_exuviae_only_analysis() 