import fiftyone as fo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

def analyze_position_impact():
    """
    Analyze how prawn position in the image affects measurement errors.
    Extracts keypoint coordinates and correlates them with measurement accuracy.
    """
    
    # Load the dataset
    dataset = fo.load_dataset("prawn_keypoints")
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Load the measurement results
    measurement_df = pd.read_csv("complete_detailed_results_table.csv")
    print(f"Loaded measurement data with {len(measurement_df)} measurements")
    
    # Prepare data for analysis
    position_data = []
    
    for sample in dataset:
        if sample.detections is None:
            continue
            
        for detection in sample.detections.detections:
            # Check if this detection has keypoints
            if not hasattr(detection, 'attributes') or 'keypoints' not in detection.attributes:
                continue
                
            keypoints_dict = detection.attributes['keypoints']
            
            # Extract keypoint coordinates (normalized 0-1)
            try:
                start_carapace = keypoints_dict['start_carapace']
                eyes = keypoints_dict['eyes']
                rostrum = keypoints_dict['rostrum']
                tail = keypoints_dict['tail']
                
                # Skip if any keypoint is NaN (at image edge)
                if (np.isnan(start_carapace[0]) or np.isnan(start_carapace[1]) or
                    np.isnan(eyes[0]) or np.isnan(eyes[1]) or
                    np.isnan(rostrum[0]) or np.isnan(rostrum[1]) or
                    np.isnan(tail[0]) or np.isnan(tail[1])):
                    continue
                    
            except (KeyError, TypeError):
                continue
            
            # Calculate center position of the prawn (average of all keypoints)
            center_x = (start_carapace[0] + eyes[0] + rostrum[0] + tail[0]) / 4
            center_y = (start_carapace[1] + eyes[1] + rostrum[1] + tail[1]) / 4
            
            # Calculate distance from image center (0.5, 0.5)
            distance_from_center = math.sqrt((center_x - 0.5)**2 + (center_y - 0.5)**2)
            
            # Calculate distance from image edges
            distance_from_left = center_x
            distance_from_right = 1 - center_x
            distance_from_top = center_y
            distance_from_bottom = 1 - center_y
            
            # Find minimum distance to any edge
            min_edge_distance = min(distance_from_left, distance_from_right, 
                                  distance_from_top, distance_from_bottom)
            
            # Get bounding box for additional position metrics
            bbox = detection.bounding_box
            if bbox is not None:
                bbox_center_x = bbox[0] + bbox[2]/2
                bbox_center_y = bbox[1] + bbox[3]/2
                bbox_area = bbox[2] * bbox[3]  # Normalized area
            else:
                bbox_center_x = center_x
                bbox_center_y = center_y
                bbox_area = None
            
            # Create position data entry
            position_entry = {
                'image_name': sample.filename,
                'center_x': center_x,
                'center_y': center_y,
                'distance_from_center': distance_from_center,
                'distance_from_left': distance_from_left,
                'distance_from_right': distance_from_right,
                'distance_from_top': distance_from_top,
                'distance_from_bottom': distance_from_bottom,
                'min_edge_distance': min_edge_distance,
                'bbox_center_x': bbox_center_x,
                'bbox_center_y': bbox_center_y,
                'bbox_area': bbox_area,
                # Keypoint-specific positions
                'start_carapace_x': start_carapace[0],
                'start_carapace_y': start_carapace[1],
                'eyes_x': eyes[0],
                'eyes_y': eyes[1],
                'rostrum_x': rostrum[0],
                'rostrum_y': rostrum[1],
                'tail_x': tail[0],
                'tail_y': tail[1]
            }
            
            position_data.append(position_entry)
    
    # Convert to DataFrame
    position_df = pd.DataFrame(position_data)
    print(f"Extracted position data for {len(position_df)} detections")
    
    # Merge with measurement data
    merged_df = pd.merge(position_df, measurement_df, on='image_name', how='inner')
    print(f"Merged data contains {len(merged_df)} measurements")
    
    # Analyze correlations between position and measurement errors
    print("\n=== POSITION IMPACT ANALYSIS ===")
    
    # Position metrics to analyze
    position_metrics = [
        'center_x', 'center_y', 'distance_from_center', 'min_edge_distance',
        'distance_from_left', 'distance_from_right', 'distance_from_top', 'distance_from_bottom'
    ]
    
    # Error metrics to analyze
    error_metrics = [
        'individual_mae', 'individual_mape', 'pixel_difference_pct', 'measurement_difference_pct', 'scale_impact_ratio_rho'
    ]
    
    # Calculate correlations
    correlation_results = []
    
    for pos_metric in position_metrics:
        for error_metric in error_metrics:
            # Remove NaN values for correlation calculation
            valid_data = merged_df[[pos_metric, error_metric]].dropna()
            
            if len(valid_data) > 10:  # Need sufficient data points
                correlation, p_value = stats.pearsonr(valid_data[pos_metric], valid_data[error_metric])
                
                correlation_results.append({
                    'position_metric': pos_metric,
                    'error_metric': error_metric,
                    'correlation': correlation,
                    'p_value': p_value,
                    'n_samples': len(valid_data)
                })
    
    # Create correlation summary
    correlation_df = pd.DataFrame(correlation_results)
    
    # Filter significant correlations (p < 0.05)
    significant_correlations = correlation_df[correlation_df['p_value'] < 0.05].sort_values('correlation', key=abs, ascending=False)
    
    print("\n=== SIGNIFICANT CORRELATIONS (p < 0.05) ===")
    if len(significant_correlations) > 0:
        for _, row in significant_correlations.head(10).iterrows():
            print(f"{row['position_metric']} vs {row['error_metric']}: r={row['correlation']:.3f}, p={row['p_value']:.4f}, n={row['n_samples']}")
    else:
        print("No significant correlations found.")
    
    # Analyze by image regions
    print("\n=== ANALYSIS BY IMAGE REGIONS ===")
    
    # Define image regions
    merged_df['image_region'] = 'center'
    merged_df.loc[merged_df['center_x'] < 0.33, 'image_region'] = 'left'
    merged_df.loc[merged_df['center_x'] > 0.67, 'image_region'] = 'right'
    merged_df.loc[merged_df['center_y'] < 0.33, 'image_region'] = 'top'
    merged_df.loc[merged_df['center_y'] > 0.67, 'image_region'] = 'bottom'
    
    # Analyze errors by region
    region_analysis = merged_df.groupby('image_region')[error_metrics].agg(['mean', 'std', 'count'])
    print("\nError metrics by image region:")
    print(region_analysis)
    
    # Analyze by distance from center
    print("\n=== ANALYSIS BY DISTANCE FROM CENTER ===")
    
    # Create distance bins
    merged_df['distance_bin'] = pd.cut(merged_df['distance_from_center'], 
                                      bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5], 
                                      labels=['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5'])
    
    distance_analysis = merged_df.groupby('distance_bin')[error_metrics].agg(['mean', 'std', 'count'])
    print("\nError metrics by distance from center:")
    print(distance_analysis)
    
    # Analyze by edge proximity
    print("\n=== ANALYSIS BY EDGE PROXIMITY ===")
    
    # Create edge distance bins
    merged_df['edge_bin'] = pd.cut(merged_df['min_edge_distance'], 
                                  bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5], 
                                  labels=['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5'])
    
    edge_analysis = merged_df.groupby('edge_bin')[error_metrics].agg(['mean', 'std', 'count'])
    print("\nError metrics by distance from image edge:")
    print(edge_analysis)
    
    # Create visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Position Impact on Measurement Errors', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Distance from center vs MAE
    axes[0, 0].scatter(merged_df['distance_from_center'], merged_df['individual_mae'], alpha=0.6)
    axes[0, 0].set_xlabel('Distance from Image Center')
    axes[0, 0].set_ylabel('Mean Absolute Error (mm)')
    axes[0, 0].set_title('Distance from Center vs MAE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scatter plot: Edge distance vs MAE
    axes[0, 1].scatter(merged_df['min_edge_distance'], merged_df['individual_mae'], alpha=0.6)
    axes[0, 1].set_xlabel('Distance from Image Edge')
    axes[0, 1].set_ylabel('Mean Absolute Error (mm)')
    axes[0, 1].set_title('Edge Distance vs MAE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Scatter plot: X position vs MAE
    axes[0, 2].scatter(merged_df['center_x'], merged_df['individual_mae'], alpha=0.6)
    axes[0, 2].set_xlabel('X Position (normalized)')
    axes[0, 2].set_ylabel('Mean Absolute Error (mm)')
    axes[0, 2].set_title('X Position vs MAE')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Box plot: MAE by image region
    region_data = [merged_df[merged_df['image_region'] == region]['individual_mae'].dropna() 
                  for region in ['left', 'center', 'right', 'top', 'bottom']]
    region_labels = ['Left', 'Center', 'Right', 'Top', 'Bottom']
    axes[1, 0].boxplot(region_data, labels=region_labels)
    axes[1, 0].set_ylabel('Mean Absolute Error (mm)')
    axes[1, 0].set_title('MAE by Image Region')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. Box plot: MAE by distance from center
    distance_data = [merged_df[merged_df['distance_bin'] == bin_val]['individual_mae'].dropna() 
                    for bin_val in merged_df['distance_bin'].cat.categories]
    axes[1, 1].boxplot(distance_data, labels=merged_df['distance_bin'].cat.categories)
    axes[1, 1].set_xlabel('Distance from Center')
    axes[1, 1].set_ylabel('Mean Absolute Error (mm)')
    axes[1, 1].set_title('MAE by Distance from Center')
    
    # 6. Heatmap of correlations
    if len(correlation_df) > 0:
        # Create pivot table for heatmap
        pivot_data = correlation_df.pivot(index='position_metric', columns='error_metric', values='correlation')
        sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Position-Error Correlations')
    else:
        axes[1, 2].text(0.5, 0.5, 'No correlation data available', ha='center', va='center')
        axes[1, 2].set_title('Position-Error Correlations')
    
    plt.tight_layout()
    plt.savefig('position_impact_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to: position_impact_analysis.png")
    
    # Save detailed results
    results_summary = {
        'correlation_analysis': correlation_df,
        'region_analysis': region_analysis,
        'distance_analysis': distance_analysis,
        'edge_analysis': edge_analysis,
        'merged_data': merged_df
    }
    
    # Save to CSV
    merged_df.to_csv('position_impact_detailed_results.csv', index=False)
    correlation_df.to_csv('position_error_correlations.csv', index=False)
    
    print("\n=== SUMMARY ===")
    print(f"Total measurements analyzed: {len(merged_df)}")
    print(f"Significant correlations found: {len(significant_correlations)}")
    
    if len(significant_correlations) > 0:
        strongest_corr = significant_correlations.iloc[0]
        print(f"Strongest correlation: {strongest_corr['position_metric']} vs {strongest_corr['error_metric']} (r={strongest_corr['correlation']:.3f})")
    
    print("\nFiles saved:")
    print("- position_impact_analysis.png: Visualization plots")
    print("- position_impact_detailed_results.csv: Complete dataset with position data")
    print("- position_error_correlations.csv: Correlation analysis results")
    
    return results_summary

if __name__ == "__main__":
    results = analyze_position_impact() 