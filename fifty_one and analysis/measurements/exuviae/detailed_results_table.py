import pandas as pd
import numpy as np

def create_detailed_results_table():
    """
    Create a comprehensive table showing all measurements with detailed information
    to help identify outliers in the scale impact ratio analysis.
    """
    
    print("📋 DETAILED RESULTS TABLE - ALL MEASUREMENTS")
    print("=" * 120)
    
    # Load the analysis results
    df = pd.read_csv('spreadsheet_files/scale_impact_analysis_results.csv')
    
    # Sort by pond type, size, and scale impact ratio for better analysis
    df_sorted = df.sort_values(['pond_type', 'manual_size', 'scale_impact_ratio_rho'])
    
    # Add index for reference
    df_sorted = df_sorted.reset_index(drop=True)
    df_sorted['index'] = df_sorted.index + 1
    
    print(f"📊 Total measurements: {len(df_sorted)}")
    print(f"📊 Filtered measurements (total_length >= 100mm): {len(df_sorted)}")
    print()
    
    # Create the detailed table
    print("📋 COMPREHENSIVE MEASUREMENT TABLE:")
    print("-" * 120)
    
    # Print header
    header = (
        f"{'IDX':<3} {'IMAGE_NAME':<35} {'POND':<6} {'SIZE':<5} "
        f"{'PRED_LEN':<8} {'REAL_LEN':<8} {'MAE':<7} {'MAPE':<7} "
        f"{'PX_TOTAL':<8} {'PX_SHAI':<8} {'SCALE_ρ':<8} {'IoU':<6}"
    )
    print(header)
    print("-" * 120)
    
    # Print each measurement
    for _, row in df_sorted.iterrows():
        # Calculate individual MAE and MAPE
        mae = abs(row['calculated_total_length_mm'] - row['real_length'])
        mape = (mae / row['real_length']) * 100
        
        # Format the row
        row_str = (
            f"{row['index']:<3} "
            f"{row['image_name'][:35]:<35} "
            f"{row['pond_type']:<6} "
            f"{row['manual_size']:<5} "
            f"{row['calculated_total_length_mm']:.1f}mm{'':<2} "
            f"{row['real_length']:.0f}mm{'':<4} "
            f"{mae:.1f}mm{'':<2} "
            f"{mape:.1f}%{'':<3} "
            f"{row['total_length_pixels']:.0f}px{'':<2} "
            f"{row['shai_length']:.0f}px{'':<2} "
            f"{row['scale_impact_ratio_rho']:.3f}{'':<3} "
            f"{row['shai_iou']:.3f}"
        )
        print(row_str)
    
    print("-" * 120)
    
    # Summary statistics by partition
    print("\n📊 OUTLIER ANALYSIS BY PARTITION:")
    print("=" * 80)
    
    for pond_type in ['Circle', 'Square']:
        for size in ['big', 'small']:
            partition_data = df_sorted[(df_sorted['pond_type'] == pond_type) & 
                                     (df_sorted['manual_size'] == size)]
            
            if len(partition_data) > 0:
                rho_values = partition_data['scale_impact_ratio_rho']
                
                print(f"\n{pond_type.upper()} POND - {size.upper()} PRAWNS ({len(partition_data)} measurements):")
                print(f"  Mean ρ: {rho_values.mean():.4f}")
                print(f"  Median ρ: {rho_values.median():.4f}")
                print(f"  Std Dev ρ: {rho_values.std():.4f}")
                print(f"  Min ρ: {rho_values.min():.4f}")
                print(f"  Max ρ: {rho_values.max():.4f}")
                print(f"  Q1 ρ: {rho_values.quantile(0.25):.4f}")
                print(f"  Q3 ρ: {rho_values.quantile(0.75):.4f}")
                print(f"  IQR ρ: {rho_values.quantile(0.75) - rho_values.quantile(0.25):.4f}")
                
                # Identify potential outliers using IQR method
                Q1 = rho_values.quantile(0.25)
                Q3 = rho_values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = partition_data[(rho_values < lower_bound) | (rho_values > upper_bound)]
                
                if len(outliers) > 0:
                    print(f"  🚨 POTENTIAL OUTLIERS ({len(outliers)} measurements):")
                    for _, outlier in outliers.iterrows():
                        print(f"    - {outlier['image_name'][:30]}: ρ={outlier['scale_impact_ratio_rho']:.4f}")
                else:
                    print(f"  ✅ No outliers detected using IQR method")
    
    # Save detailed results
    print(f"\n💾 SAVING DETAILED RESULTS...")
    
    # Prepare detailed CSV
    detailed_df = df_sorted.copy()
    detailed_df['individual_mae'] = abs(detailed_df['calculated_total_length_mm'] - detailed_df['real_length'])
    detailed_df['individual_mape'] = (detailed_df['individual_mae'] / detailed_df['real_length']) * 100
    
    # Reorder columns for better readability
    columns_order = [
        'index', 'image_name', 'pond_type', 'manual_size',
        'calculated_total_length_mm', 'real_length', 'individual_mae', 'individual_mape',
        'total_length_pixels', 'shai_length', 'pixel_difference_pct', 'measurement_difference_pct',
        'scale_impact_ratio_rho', 'shai_iou'
    ]
    
    detailed_df = detailed_df[columns_order]
    detailed_df.to_csv('spreadsheet_files/detailed_measurements_table.csv', index=False)
    
    print(f"📊 Detailed table saved as 'spreadsheet_files/detailed_measurements_table.csv'")
    
    # Create summary of extreme values
    print(f"\n🔍 EXTREME VALUES ANALYSIS:")
    print("=" * 80)
    
    # Highest ρ values
    highest_rho = df_sorted.nlargest(5, 'scale_impact_ratio_rho')
    print(f"\n📈 TOP 5 HIGHEST ρ VALUES:")
    for _, row in highest_rho.iterrows():
        print(f"  {row['image_name'][:30]:<30} | {row['pond_type']}-{row['manual_size']:<5} | ρ={row['scale_impact_ratio_rho']:.4f}")
    
    # Lowest ρ values
    lowest_rho = df_sorted.nsmallest(5, 'scale_impact_ratio_rho')
    print(f"\n📉 TOP 5 LOWEST ρ VALUES:")
    for _, row in lowest_rho.iterrows():
        print(f"  {row['image_name'][:30]:<30} | {row['pond_type']}-{row['manual_size']:<5} | ρ={row['scale_impact_ratio_rho']:.4f}")
    
    # Highest pixel errors
    highest_px_error = df_sorted.nlargest(5, 'pixel_difference_pct')
    print(f"\n📊 TOP 5 HIGHEST PIXEL ERRORS:")
    for _, row in highest_px_error.iterrows():
        print(f"  {row['image_name'][:30]:<30} | {row['pond_type']}-{row['manual_size']:<5} | Δpx={row['pixel_difference_pct']:.2f}%")
    
    # Highest measurement errors
    highest_mm_error = df_sorted.nlargest(5, 'measurement_difference_pct')
    print(f"\n📏 TOP 5 HIGHEST MEASUREMENT ERRORS:")
    for _, row in highest_mm_error.iterrows():
        print(f"  {row['image_name'][:30]:<30} | {row['pond_type']}-{row['manual_size']:<5} | Δmm={row['measurement_difference_pct']:.2f}%")

if __name__ == "__main__":
    create_detailed_results_table() 