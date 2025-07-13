import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_scale_impact_ratio():
    """
    Analyze Scale Impact Ratio (ρ) from the merged manual and Shai keypoints CSV.
    ρ = Δmm% / Δpx% where:
    - Δmm% = percentage error in mm measurements vs real length
    - Δpx% = percentage error in pixel measurements vs Shai's measurements
    """
    
    print("⚖️ SCALE IMPACT RATIO (ρ) ANALYSIS")
    print("=" * 80)
    print("📊 Computing ρ = Δmm% / Δpx% for Circle vs Square ponds...")
    print("=" * 80)
    
    # Load the merged dataset
    df = pd.read_csv("spreadsheet_files/merged_manual_shai_keypoints.csv")
    print(f"✅ Loaded {len(df)} measurements from merged CSV")
    
    # Add pond type based on image name
    df['pond_type'] = df['image_name'].apply(lambda x: 'Circle' if 'GX010191' in x else 'Square')
    
    # Set real reference lengths based on manual size classification
    df['real_length'] = df['manual_size'].map({'big': 180, 'small': 145})
    
    # Calculate total length in pixels from keypoints (rostrum to tail distance)
    img_width_px = 5312
    img_height_px = 2988
    
    # Convert normalized coordinates to pixels and calculate distance
    df['rostrum_x_px'] = df['rostrum_x'] * img_width_px
    df['rostrum_y_px'] = df['rostrum_y'] * img_height_px
    df['tail_x_px'] = df['tail_x'] * img_width_px
    df['tail_y_px'] = df['tail_y'] * img_height_px
    
    df['total_length_pixels'] = np.sqrt((df['rostrum_x_px'] - df['tail_x_px'])**2 + 
                                       (df['rostrum_y_px'] - df['tail_y_px'])**2)
    
    # Calculate pixel percentage differences (Δpx%)
    # Compare calculated total length in pixels vs Shai's pixel measurements
    df['pixel_difference_pct'] = abs(df['total_length_pixels'] - df['shai_length']) / df['shai_length'] * 100
    
    # Calculate measurement percentage differences after scaling (Δmm%)
    # Compare calculated total length in mm vs real reference length
    df['measurement_difference_pct'] = abs(df['calculated_total_length_mm'] - df['real_length']) / df['real_length'] * 100
    
    # Calculate Scale Impact Ratio: ρ = Δmm% / Δpx%
    # Handle division by zero cases
    df['scale_impact_ratio_rho'] = np.where(
        df['pixel_difference_pct'] != 0,
        df['measurement_difference_pct'] / df['pixel_difference_pct'],
        np.nan  # Set to NaN when pixel difference is zero
    )
    
    # Filter out invalid measurements (NaN values) and sizes smaller than 100
    valid_df = df.dropna(subset=['scale_impact_ratio_rho', 'total_length_pixels', 'calculated_total_length_mm'])
    valid_df = valid_df[valid_df['calculated_total_length_mm'] >= 100]
    
    print(f"📊 Valid measurements for analysis: {len(valid_df)} (filtered: total_length >= 100mm)")
    print(f"📊 Using percentage errors: ρ = (Δmm%/real_length) / (Δpx%/shai_length)")
    
    # Statistical Analysis by Pond Type AND Size (Partitioned Analysis)
    print(f"\n📈 STATISTICAL ANALYSIS BY POND TYPE AND SIZE PARTITIONS")
    print("=" * 80)
    
    partition_stats = {}
    
    # Analyze each combination of pond type and size
    for pond_type in ['Circle', 'Square']:
        for size in ['big', 'small']:
            partition_key = f"{pond_type}_{size}"
            partition_data = valid_df[(valid_df['pond_type'] == pond_type) & (valid_df['manual_size'] == size)]
            rho_values = partition_data['scale_impact_ratio_rho'].dropna()
            valid_measurements = len(rho_values)
        
            if valid_measurements > 0:
                # Calculate MAE and MAPE for total length measurements
                mae_total = abs(partition_data['calculated_total_length_mm'] - partition_data['real_length']).mean()
                mape_total = (abs(partition_data['calculated_total_length_mm'] - partition_data['real_length']) / partition_data['real_length'] * 100).mean()
                
                # Calculate statistics
                stats_dict = {
                    'pond_type': pond_type,
                    'size': size,
                    'count': valid_measurements,
                    'mean': rho_values.mean(),
                    'median': rho_values.median(),
                    'std': rho_values.std(),
                    'min': rho_values.min(),
                    'max': rho_values.max(),
                    'q25': rho_values.quantile(0.25),
                    'q75': rho_values.quantile(0.75),
                    'iqr': rho_values.quantile(0.75) - rho_values.quantile(0.25),
                    'mae_total_length': mae_total,
                    'mape_total_length': mape_total
                }
                
                # Scaling behavior classification
                neutral_scaling = (abs(rho_values - 1) < 0.1).sum()
                amplifying_scaling = (rho_values > 1.1).sum()
                compressing_scaling = (rho_values < 0.9).sum()
                
                stats_dict.update({
                    'neutral_count': neutral_scaling,
                    'amplifying_count': amplifying_scaling,
                    'compressing_count': compressing_scaling,
                    'neutral_pct': neutral_scaling / valid_measurements * 100,
                    'amplifying_pct': amplifying_scaling / valid_measurements * 100,
                    'compressing_pct': compressing_scaling / valid_measurements * 100
                })
                
                partition_stats[partition_key] = stats_dict
                
                # Display statistics for each partition
                pond_icon = "🔵" if pond_type == "Circle" else "🔲"
                size_icon = "🦐" if size == "big" else "🦐"
                print(f"\n{pond_icon} {pond_type.upper()} POND - {size_icon} {size.upper()} PRAWNS:")
                print(f"   📊 Valid measurements: {valid_measurements}")
                print(f"   📊 **Mean ρ: {stats_dict['mean']:.4f}** (primary metric)")
                print(f"   📊 Median ρ: {stats_dict['median']:.4f}")
                print(f"   📊 IQR: {stats_dict['iqr']:.4f}")
                print(f"   📊 Range: [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]")
                print(f"   📊 MAE (Total Length): {stats_dict['mae_total_length']:.2f}mm")
                print(f"   📊 MAPE (Total Length): {stats_dict['mape_total_length']:.2f}%")
                
                print(f"\n   ⚖️ SCALING BEHAVIOR:")
                print(f"   🔄 Neutral (|ρ-1| < 0.1): {neutral_scaling} ({stats_dict['neutral_pct']:.1f}%)")
                print(f"   📈 Amplifying (ρ > 1.1): {amplifying_scaling} ({stats_dict['amplifying_pct']:.1f}%)")
                print(f"   📉 Compressing (ρ < 0.9): {compressing_scaling} ({stats_dict['compressing_pct']:.1f}%)")
                
                # Interpretation based on mean
                mean_rho = stats_dict['mean']
                print(f"\n   🔍 INTERPRETATION (based on mean ρ):")
                if abs(mean_rho - 1) < 0.1:
                    print(f"   ✅ NEUTRAL scaling (mean ρ ≈ 1): Percentage errors preserved")
                elif mean_rho > 1.1:
                    print(f"   ⚠️ AMPLIFYING scaling (mean ρ > 1): Errors magnified by {mean_rho:.1f}x")
                elif mean_rho < 0.9:
                    print(f"   ⚠️ COMPRESSING scaling (mean ρ < 1): Errors reduced to {mean_rho:.1f}x")
                else:
                    print(f"   🔍 MIXED behavior (mean ρ = {mean_rho:.3f}): Variable scaling impact")
            else:
                print(f"\n{pond_icon} {pond_type.upper()} POND - {size.upper()} PRAWNS: No valid measurements")
    
    # Comparative Analysis by Partitions
    print(f"\n🔄 COMPARATIVE ANALYSIS: Partitioned Results")
    print("=" * 80)
    
    # Compare by size within each pond type
    for size in ['big', 'small']:
        circle_key = f"Circle_{size}"
        square_key = f"Square_{size}"
        
        circle_stats = partition_stats.get(circle_key, {})
        square_stats = partition_stats.get(square_key, {})
        
        if circle_stats and square_stats:
            print(f"\n🦐 {size.upper()} PRAWNS - Circle vs Square:")
            print(f"   🔵 Circle: mean ρ={circle_stats['mean']:.4f}, MAE={circle_stats['mae_total_length']:.2f}mm, MAPE={circle_stats['mape_total_length']:.2f}%")
            print(f"   🔲 Square: mean ρ={square_stats['mean']:.4f}, MAE={square_stats['mae_total_length']:.2f}mm, MAPE={square_stats['mape_total_length']:.2f}%")
            
            # Determine which pond has better scaling behavior for this size
            circle_deviation = abs(circle_stats['mean'] - 1)
            square_deviation = abs(square_stats['mean'] - 1)
            
            if circle_deviation < square_deviation:
                print(f"   🏆 Circle pond has more neutral scaling for {size} prawns")
            elif square_deviation < circle_deviation:
                print(f"   🏆 Square pond has more neutral scaling for {size} prawns")
            else:
                print(f"   ⚖️ Similar scaling behavior for {size} prawns")
    
    # Compare by pond type within each size
    for pond_type in ['Circle', 'Square']:
        big_key = f"{pond_type}_big"
        small_key = f"{pond_type}_small"
        
        big_stats = partition_stats.get(big_key, {})
        small_stats = partition_stats.get(small_key, {})
        
        if big_stats and small_stats:
            pond_icon = "🔵" if pond_type == "Circle" else "🔲"
            print(f"\n{pond_icon} {pond_type.upper()} POND - Big vs Small:")
            print(f"   🦐 Big: mean ρ={big_stats['mean']:.4f}, MAE={big_stats['mae_total_length']:.2f}mm, MAPE={big_stats['mape_total_length']:.2f}%")
            print(f"   🦐 Small: mean ρ={small_stats['mean']:.4f}, MAE={small_stats['mae_total_length']:.2f}mm, MAPE={small_stats['mape_total_length']:.2f}%")
            
            # Determine which size has better scaling behavior in this pond
            big_deviation = abs(big_stats['mean'] - 1)
            small_deviation = abs(small_stats['mean'] - 1)
            
            if big_deviation < small_deviation:
                print(f"   🏆 Big prawns have more neutral scaling in {pond_type} pond")
            elif small_deviation < big_deviation:
                print(f"   🏆 Small prawns have more neutral scaling in {pond_type} pond")
            else:
                print(f"   ⚖️ Similar scaling behavior for both sizes in {pond_type} pond")
    
    # Display sample calculations by pond type
    print(f"\n📋 SAMPLE CALCULATIONS BY POND TYPE:")
    print("-" * 60)
    
    sample_cols = ['image_name', 'pond_type', 'manual_size', 'shai_length', 'total_length_pixels', 
                   'calculated_total_length_mm', 'real_length', 'pixel_difference_pct', 
                   'measurement_difference_pct', 'scale_impact_ratio_rho']
    
    print("\nFirst 15 measurements:")
    print(valid_df[sample_cols].head(15).to_string(index=False))
    

    
    # Create visualizations
    create_visualizations(valid_df)
    
    # Create easy-to-copy table
    create_summary_table(partition_stats)
    
    # Save results
    save_results(valid_df, partition_stats)
    
    return valid_df, partition_stats

def create_visualizations(df):
    """Create visualizations for the scale impact ratio analysis"""
    
    print(f"\n📊 CREATING VISUALIZATIONS...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Scale Impact Ratio (ρ) Analysis: Circle vs Square Ponds', fontsize=16, fontweight='bold')
    
    # 1. Box plot by pond type
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='pond_type', y='scale_impact_ratio_rho', ax=ax1)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect scaling (ρ=1)')
    ax1.set_title('Scale Impact Ratio by Pond Type')
    ax1.set_ylabel('Scale Impact Ratio (ρ)')
    ax1.legend()
    
    # 2. Histogram by pond type
    ax2 = axes[0, 1]
    for pond_type in ['Circle', 'Square']:
        pond_data = df[df['pond_type'] == pond_type]['scale_impact_ratio_rho']
        ax2.hist(pond_data, alpha=0.7, label=f'{pond_type} (n={len(pond_data)})', bins=15)
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Perfect scaling (ρ=1)')
    ax2.set_title('Distribution of Scale Impact Ratios')
    ax2.set_xlabel('Scale Impact Ratio (ρ)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # 3. Scatter plot: pixel error vs measurement error
    ax3 = axes[1, 0]
    colors = {'Circle': 'blue', 'Square': 'orange'}
    for pond_type in ['Circle', 'Square']:
        pond_data = df[df['pond_type'] == pond_type]
        ax3.scatter(pond_data['pixel_difference_pct'], pond_data['measurement_difference_pct'], 
                   c=colors[pond_type], alpha=0.7, label=f'{pond_type} (n={len(pond_data)})')
    
    # Add perfect scaling line (y = x)
    max_val = max(df['pixel_difference_pct'].max(), df['measurement_difference_pct'].max())
    ax3.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Perfect scaling (ρ=1)')
    ax3.set_xlabel('Pixel Error (%)')
    ax3.set_ylabel('Measurement Error (%)')
    ax3.set_title('Pixel Error vs Measurement Error')
    ax3.legend()
    
    # 4. Box plot by size classification
    ax4 = axes[1, 1]
    sns.boxplot(data=df, x='manual_size', y='scale_impact_ratio_rho', hue='pond_type', ax=ax4)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect scaling (ρ=1)')
    ax4.set_title('Scale Impact Ratio by Size and Pond Type')
    ax4.set_ylabel('Scale Impact Ratio (ρ)')
    ax4.set_xlabel('Manual Size Classification')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('scale_impact_ratio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualizations saved as 'scale_impact_ratio_analysis.png'")

def create_summary_table(partition_stats):
    """Create an easy-to-copy summary table of key results"""
    
    print(f"\n📋 EASY-TO-COPY SUMMARY TABLE")
    print("=" * 100)
    
    # Create table headers
    headers = ["Partition", "Count", "Mean ρ", "MAE (mm)", "MAPE (%)", "Neutral %", "Amplifying %", "Compressing %"]
    
    # Create table data
    table_data = []
    
    # Define order for consistent display
    partition_order = ["Circle_big", "Circle_small", "Square_big", "Square_small"]
    
    for partition_key in partition_order:
        if partition_key in partition_stats:
            stats = partition_stats[partition_key]
            pond_type = stats['pond_type']
            size = stats['size']
            
            row = [
                f"{pond_type}-{size}",
                f"{stats['count']}",
                f"{stats['mean']:.4f}",
                f"{stats['mae_total_length']:.2f}",
                f"{stats['mape_total_length']:.2f}",
                f"{stats['neutral_pct']:.1f}",
                f"{stats['amplifying_pct']:.1f}",
                f"{stats['compressing_pct']:.1f}"
            ]
            table_data.append(row)
    
    # Calculate column widths
    col_widths = []
    for i in range(len(headers)):
        max_width = len(headers[i])
        for row in table_data:
            max_width = max(max_width, len(row[i]))
        col_widths.append(max_width + 2)  # Add padding
    
    # Print table
    def print_row(row_data, widths):
        row_str = "|"
        for i, cell in enumerate(row_data):
            row_str += f" {cell:<{widths[i]-1}}|"
        print(row_str)
    
    def print_separator(widths):
        sep_str = "|"
        for width in widths:
            sep_str += "-" * width + "|"
        print(sep_str)
    
    # Print the table
    print_separator(col_widths)
    print_row(headers, col_widths)
    print_separator(col_widths)
    
    for row in table_data:
        print_row(row, col_widths)
    
    print_separator(col_widths)
    
    print(f"\n📋 COPY-PASTE FRIENDLY VERSION (Tab-separated):")
    print("-" * 80)
    
    # Tab-separated version for easy copying
    print("\t".join(headers))
    for row in table_data:
        print("\t".join(row))
    
    print(f"\n📋 CSV FORMAT:")
    print("-" * 80)
    
    # CSV version
    print(",".join(headers))
    for row in table_data:
        print(",".join(row))
    
    # Save table to file
    with open('summary_table.txt', 'w') as f:
        f.write("SCALE IMPACT RATIO ANALYSIS - SUMMARY TABLE\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("TAB-SEPARATED FORMAT:\n")
        f.write("\t".join(headers) + "\n")
        for row in table_data:
            f.write("\t".join(row) + "\n")
        
        f.write("\nCSV FORMAT:\n")
        f.write(",".join(headers) + "\n")
        for row in table_data:
            f.write(",".join(row) + "\n")
        
        f.write("\nKEY INTERPRETATIONS:\n")
        f.write("- ρ (Scale Impact Ratio): Δmm% / Δpx%\n")
        f.write("- Ideal ρ ≈ 1.0 (neutral scaling)\n")
        f.write("- ρ > 1.1: Amplifying (errors magnified)\n")
        f.write("- ρ < 0.9: Compressing (errors reduced)\n")
        f.write("- MAE: Mean Absolute Error in mm\n")
        f.write("- MAPE: Mean Absolute Percentage Error\n")
    
    print(f"\n📊 Summary table saved as 'summary_table.txt'")

def save_results(df, partition_stats):
    """Save the analysis results to CSV files"""
    
    print(f"\n💾 SAVING RESULTS...")
    
    # Save the enhanced dataset
    output_df = df[['image_name', 'pond_type', 'manual_size', 'shai_length', 'total_length_pixels',
                   'calculated_total_length_mm', 'real_length', 'pixel_difference_pct', 
                   'measurement_difference_pct', 'scale_impact_ratio_rho', 'shai_iou']]
    
    output_df.to_csv('spreadsheet_files/scale_impact_analysis_results.csv', index=False)
    print(f"📊 Enhanced dataset saved as 'spreadsheet_files/scale_impact_analysis_results.csv'")
    
    # Save partition statistics
    partition_stats_df = pd.DataFrame(partition_stats).T
    partition_stats_df.to_csv('spreadsheet_files/partition_statistics_scale_impact.csv')
    print(f"📊 Partition statistics saved as 'spreadsheet_files/partition_statistics_scale_impact.csv'")
    
    # Create summary report
    summary_report = []
    summary_report.append("SCALE IMPACT RATIO (ρ) ANALYSIS SUMMARY - PARTITIONED BY POND TYPE AND SIZE")
    summary_report.append("=" * 80)
    summary_report.append(f"Total valid measurements: {len(df)} (filtered: total_length >= 100mm)")
    summary_report.append(f"Circle pond measurements: {len(df[df['pond_type'] == 'Circle'])}")
    summary_report.append(f"Square pond measurements: {len(df[df['pond_type'] == 'Square'])}")
    summary_report.append("")
    summary_report.append("METRICS:")
    summary_report.append("- ρ (Scale Impact Ratio): Δmm% / Δpx%")
    summary_report.append("- MAE (Mean Absolute Error): |calculated - real| in mm")
    summary_report.append("- MAPE (Mean Absolute Percentage Error): |calculated - real| / real * 100%")
    summary_report.append("")
    
    for partition_key, stats in partition_stats.items():
        pond_type = stats['pond_type']
        size = stats['size']
        summary_report.append(f"{pond_type.upper()} POND - {size.upper()} PRAWNS:")
        summary_report.append(f"  Count: {stats['count']}")
        summary_report.append(f"  Mean ρ: {stats['mean']:.4f}")
        summary_report.append(f"  IQR: {stats['iqr']:.4f}")
        summary_report.append(f"  MAE (Total Length): {stats['mae_total_length']:.2f}mm")
        summary_report.append(f"  MAPE (Total Length): {stats['mape_total_length']:.2f}%")
        summary_report.append(f"  Neutral scaling: {stats['neutral_pct']:.1f}%")
        summary_report.append(f"  Amplifying scaling: {stats['amplifying_pct']:.1f}%")
        summary_report.append(f"  Compressing scaling: {stats['compressing_pct']:.1f}%")
        summary_report.append("")
    
    with open('scale_impact_analysis_summary.txt', 'w') as f:
        f.write('\n'.join(summary_report))
    
    print(f"📊 Summary report saved as 'scale_impact_analysis_summary.txt'")

if __name__ == "__main__":
    df, stats = analyze_scale_impact_ratio() 