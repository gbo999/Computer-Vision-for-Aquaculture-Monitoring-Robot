import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def iterative_outlier_analysis():
    """
    Replicates the exact iterative outlier removal process from the original analysis.
    Shows that 8 measurements are removed through iterative IQR filtering.
    """
    
    print("=" * 80)
    print("ITERATIVE OUTLIER REMOVAL ANALYSIS")
    print("=" * 80)
    print("🔍 Replicating the exact process from small_exuviae_only_analysis.py")
    print("📊 Shows iterative outlier removal: total_error → carapace_error → pixel_error")
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
    
    # Remove outliers using IQR method (same as original)
    def remove_outliers_iqr(data, column, factor=1.5):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # Replicate the exact iterative process from the original script
    print(f"\n🔄 ITERATIVE OUTLIER REMOVAL PROCESS:")
    print("=" * 60)
    
    small_df_clean = small_df.copy()
    removed_images = []
    
    # Step 1: Remove total_error_mm outliers
    print(f"Step 1: Remove total_error_mm outliers")
    print(f"   Before: {len(small_df_clean)} measurements")
    
    # Find outliers for total_error_mm
    Q1 = small_df_clean['total_error_mm'].quantile(0.25)
    Q3 = small_df_clean['total_error_mm'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    total_error_outliers = small_df_clean[(small_df_clean['total_error_mm'] < lower_bound) | 
                                         (small_df_clean['total_error_mm'] > upper_bound)]
    
    print(f"   Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   Outliers found: {len(total_error_outliers)}")
    
    for idx, row in total_error_outliers.iterrows():
        print(f"     - {row['image_name']}: {row['total_error_mm']:.2f} mm")
        removed_images.append(row['image_name'])
    
    # Remove total_error_mm outliers
    small_df_clean = remove_outliers_iqr(small_df_clean, 'total_error_mm')
    print(f"   After: {len(small_df_clean)} measurements")
    print()
    
    # Step 2: Remove carapace_error_mm outliers from remaining data
    print(f"Step 2: Remove carapace_error_mm outliers")
    print(f"   Before: {len(small_df_clean)} measurements")
    
    # Find outliers for carapace_error_mm
    Q1 = small_df_clean['carapace_error_mm'].quantile(0.25)
    Q3 = small_df_clean['carapace_error_mm'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    carapace_error_outliers = small_df_clean[(small_df_clean['carapace_error_mm'] < lower_bound) | 
                                            (small_df_clean['carapace_error_mm'] > upper_bound)]
    
    print(f"   Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   Outliers found: {len(carapace_error_outliers)}")
    
    for idx, row in carapace_error_outliers.iterrows():
        print(f"     - {row['image_name']}: {row['carapace_error_mm']:.2f} mm")
        if row['image_name'] not in removed_images:
            removed_images.append(row['image_name'])
    
    # Remove carapace_error_mm outliers
    small_df_clean = remove_outliers_iqr(small_df_clean, 'carapace_error_mm')
    print(f"   After: {len(small_df_clean)} measurements")
    print()
    
    # Step 3: Remove pixel_error outliers from remaining data
    print(f"Step 3: Remove pixel_error outliers")
    print(f"   Before: {len(small_df_clean)} measurements")
    
    # Find outliers for pixel_error
    Q1 = small_df_clean['pixel_error'].quantile(0.25)
    Q3 = small_df_clean['pixel_error'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    pixel_error_outliers = small_df_clean[(small_df_clean['pixel_error'] < lower_bound) | 
                                         (small_df_clean['pixel_error'] > upper_bound)]
    
    print(f"   Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"   Outliers found: {len(pixel_error_outliers)}")
    
    for idx, row in pixel_error_outliers.iterrows():
        print(f"     - {row['image_name']}: {row['pixel_error']:.2f} px")
        if row['image_name'] not in removed_images:
            removed_images.append(row['image_name'])
    
    # Remove pixel_error outliers
    small_df_clean = remove_outliers_iqr(small_df_clean, 'pixel_error')
    print(f"   After: {len(small_df_clean)} measurements")
    
    print(f"\n" + "=" * 80)
    print("📋 FINAL RESULTS")
    print("=" * 80)
    print(f"Original measurements: {len(small_df)}")
    print(f"Final measurements: {len(small_df_clean)}")
    print(f"Total measurements removed: {len(small_df) - len(small_df_clean)}")
    print(f"Unique images removed: {len(set(removed_images))}")
    
    print(f"\n🗑️ REMOVED IMAGES ({len(set(removed_images))} unique):")
    print("-" * 40)
    for img in sorted(set(removed_images)):
        print(f"  - {img}")
    
    # Verify this matches the original analysis
    print(f"\n✅ VERIFICATION:")
    print(f"   Expected remaining: 27")
    print(f"   Actual remaining: {len(small_df_clean)}")
    print(f"   Match: {'✅ YES' if len(small_df_clean) == 27 else '❌ NO'}")
    
    print("\n" + "=" * 80)
    print("🎯 KEY INSIGHT")
    print("=" * 80)
    print("The iterative outlier removal process removes outliers step-by-step:")
    print("1. Remove total_error_mm outliers first")
    print("2. From remaining data, remove carapace_error_mm outliers")
    print("3. From remaining data, remove pixel_error outliers")
    print("This results in 8 unique measurements being removed.")
    print("=" * 80)

if __name__ == "__main__":
    iterative_outlier_analysis() 