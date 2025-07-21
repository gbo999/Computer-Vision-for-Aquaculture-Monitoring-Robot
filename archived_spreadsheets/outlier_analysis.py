import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def outlier_analysis():
    """
    Detailed analysis of outliers removed from small exuviae measurements.
    Shows each outlier and justifies why it was removed using IQR method.
    """
    
    print("=" * 80)
    print("OUTLIER ANALYSIS - SMALL EXUVIAE MEASUREMENTS")
    print("=" * 80)
    print("🔍 Showing each outlier and justification for removal")
    print("📊 Using IQR method (Q1 - 1.5*IQR to Q3 + 1.5*IQR)")
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
    
    # Function to identify outliers using IQR method
    def identify_outliers_iqr(data, column, factor=1.5):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        inliers = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        
        return outliers, inliers, Q1, Q3, IQR, lower_bound, upper_bound
    
    # Analyze outliers for each metric
    metrics_to_check = ['total_error_mm', 'carapace_error_mm', 'pixel_error']
    
    all_outliers = []
    all_inliers = small_df.copy()
    
    for metric in metrics_to_check:
        print(f"\n" + "=" * 60)
        print(f"🔍 OUTLIER ANALYSIS: {metric.upper()}")
        print("=" * 60)
        
        outliers, inliers, Q1, Q3, IQR, lower_bound, upper_bound = identify_outliers_iqr(small_df, metric)
        
        print(f"📊 Statistics for {metric}:")
        print(f"   Q1 (25th percentile): {Q1:.2f}")
        print(f"   Q3 (75th percentile): {Q3:.2f}")
        print(f"   IQR (Q3 - Q1): {IQR:.2f}")
        print(f"   Lower bound (Q1 - 1.5*IQR): {lower_bound:.2f}")
        print(f"   Upper bound (Q3 + 1.5*IQR): {upper_bound:.2f}")
        print(f"   Total measurements: {len(small_df)}")
        print(f"   Inliers: {len(inliers)}")
        print(f"   Outliers: {len(outliers)}")
        
        if len(outliers) > 0:
            print(f"\n❌ OUTLIERS REMOVED for {metric}:")
            print("-" * 40)
            
            for idx, row in outliers.iterrows():
                print(f"   Image: {row['image_name']}")
                print(f"   Pond: {row['pond_type']}")
                print(f"   {metric}: {row[metric]:.2f}")
                print(f"   Total Length: {row['calculated_total_length_mm']:.2f} mm")
                print(f"   Carapace Length: {row['calculated_carapace_length_mm']:.2f} mm")
                print(f"   Pixel Error: {row['pixel_error']:.2f} px")
                print(f"   IoU with Shai: {row['shai_iou']:.3f}")
                print(f"   Keypoints Valid: {row['keypoints_valid_total']}/4")
                print()
                
                # Add to all outliers list
                outlier_info = {
                    'image_name': row['image_name'],
                    'pond_type': row['pond_type'],
                    'metric': metric,
                    'value': row[metric],
                    'total_length_mm': row['calculated_total_length_mm'],
                    'carapace_length_mm': row['calculated_carapace_length_mm'],
                    'pixel_error': row['pixel_error'],
                    'shai_iou': row['shai_iou'],
                    'keypoints_valid': row['keypoints_valid_total'],
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                all_outliers.append(outlier_info)
        else:
            print(f"✅ No outliers found for {metric}")
        
        # Update inliers for next iteration
        all_inliers = inliers.copy()
    
    # Show final outlier summary
    print("\n" + "=" * 80)
    print("📋 FINAL OUTLIER SUMMARY")
    print("=" * 80)
    
    if all_outliers:
        # Convert to DataFrame for better display
        outliers_df = pd.DataFrame(all_outliers)
        
        print(f"Total outliers removed: {len(all_outliers)}")
        print(f"Measurements remaining: {len(all_inliers)}")
        print(f"Original measurements: {len(small_df)}")
        
        print(f"\nOutliers by metric:")
        metric_counts = outliers_df['metric'].value_counts()
        for metric, count in metric_counts.items():
            print(f"   {metric}: {count} outliers")
        
        print(f"\nOutliers by pond type:")
        pond_counts = outliers_df['pond_type'].value_counts()
        for pond, count in pond_counts.items():
            print(f"   {pond}: {count} outliers")
        
        # Show detailed outlier table
        print(f"\n📊 DETAILED OUTLIER TABLE:")
        print("-" * 80)
        
        # Group by image to show all outliers for each image
        image_outliers = outliers_df.groupby('image_name').agg({
            'pond_type': 'first',
            'metric': lambda x: ', '.join(x),
            'value': lambda x: ', '.join([f"{v:.2f}" for v in x]),
            'total_length_mm': 'first',
            'carapace_length_mm': 'first',
            'pixel_error': 'first',
            'shai_iou': 'first',
            'keypoints_valid': 'first'
        }).reset_index()
        
        for idx, row in image_outliers.iterrows():
            print(f"Image: {row['image_name']}")
            print(f"Pond: {row['pond_type']}")
            print(f"Outlier metrics: {row['metric']}")
            print(f"Outlier values: {row['value']}")
            print(f"Total Length: {row['total_length_mm']:.2f} mm")
            print(f"Carapace Length: {row['carapace_length_mm']:.2f} mm")
            print(f"Pixel Error: {row['pixel_error']:.2f} px")
            print(f"IoU with Shai: {row['shai_iou']:.3f}")
            print(f"Keypoints Valid: {row['keypoints_valid']}/4")
            print("-" * 40)
        
        # Save outlier details to file
        outliers_df.to_csv('outlier_details.csv', index=False)
        print(f"\n💾 Outlier details saved to: outlier_details.csv")
        
    else:
        print("✅ No outliers were found in the dataset")
    
    # Justification for outlier removal
    print("\n" + "=" * 80)
    print("🎯 JUSTIFICATION FOR OUTLIER REMOVAL")
    print("=" * 80)
    print("1. **IQR Method**: Standard statistical method for outlier detection")
    print("   - Uses 25th and 75th percentiles to define normal range")
    print("   - Outliers are values outside Q1 - 1.5*IQR to Q3 + 1.5*IQR")
    print()
    print("2. **Why Remove Outliers**:")
    print("   - Outliers represent failed or unreliable measurements")
    print("   - Poor keypoint placement due to exuviae quality issues")
    print("   - Measurement errors that don't reflect true system performance")
    print("   - Ensures robust statistical analysis")
    print()
    print("3. **Impact on Analysis**:")
    print("   - More reliable error estimates")
    print("   - Better representation of system performance")
    print("   - Robust statistical conclusions")
    print("=" * 80)

if __name__ == "__main__":
    outlier_analysis() 