import fiftyone as fo
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import ast

def extract_points_from_keypoints_obj(keypoints_obj):
    """Extract points from a FiftyOne Keypoints object"""
    if not hasattr(keypoints_obj, 'keypoints') or not keypoints_obj.keypoints:
        return None
    
    # Get the first keypoint object
    kp = keypoints_obj.keypoints[0]
    if not hasattr(kp, 'points') or not kp.points:
        return None
    
    points = kp.points
    if len(points) < 4:
        return None
    
    return points

def create_position_error_heatmap():
    """
    Create a Plotly scatter plot showing prawn positions in the image with error values as colors.
    This creates a visual heatmap of measurement errors across the image.
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
        # Try to extract from detections first
        if sample.detections is not None:
            for detection in sample.detections.detections:
                # Check if detection has keypoints
                if hasattr(detection, 'keypoints') and detection.keypoints is not None:
                    points = extract_points_from_keypoints_obj(detection.keypoints)
                    if points is None:
                        continue
                    
                    # Extract the 4 keypoints: start_carapace, eyes, rostrum, tail
                    start_carapace = points[0]
                    eyes = points[1]
                    rostrum = points[2]
                    tail = points[3]
                    
                    # Skip if any keypoint is NaN
                    if (np.isnan(start_carapace[0]) or np.isnan(start_carapace[1]) or
                        np.isnan(eyes[0]) or np.isnan(eyes[1]) or
                        np.isnan(rostrum[0]) or np.isnan(rostrum[1]) or
                        np.isnan(tail[0]) or np.isnan(tail[1])):
                        continue
                    
                    # Calculate center position
                    center_x = (start_carapace[0] + eyes[0] + rostrum[0] + tail[0]) / 4
                    center_y = (start_carapace[1] + eyes[1] + rostrum[1] + tail[1]) / 4
                    
                    position_entry = {
                        'image_name': sample.filename,
                        'center_x': center_x,
                        'center_y': center_y,
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
        
        # Fallback: try sample.keypoints (if present)
        elif hasattr(sample, 'keypoints') and sample.keypoints is not None:
            points = extract_points_from_keypoints_obj(sample.keypoints)
            if points is not None:
                start_carapace = points[0]
                eyes = points[1]
                rostrum = points[2]
                tail = points[3]
                
                if (np.isnan(start_carapace[0]) or np.isnan(start_carapace[1]) or
                    np.isnan(eyes[0]) or np.isnan(eyes[1]) or
                    np.isnan(rostrum[0]) or np.isnan(rostrum[1]) or
                    np.isnan(tail[0]) or np.isnan(tail[1])):
                    continue
                
                center_x = (start_carapace[0] + eyes[0] + rostrum[0] + tail[0]) / 4
                center_y = (start_carapace[1] + eyes[1] + rostrum[1] + tail[1]) / 4
                
                position_entry = {
                    'image_name': sample.filename,
                    'center_x': center_x,
                    'center_y': center_y,
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
    
    position_df = pd.DataFrame(position_data)
    print(f"Extracted position data for {len(position_df)} detections")
    
    if len(position_df) == 0:
        print("No position data extracted. Please check the dataset structure and keypoint storage.")
        return None
    
    merged_df = pd.merge(position_df, measurement_df, on='image_name', how='inner')
    print(f"Merged data contains {len(merged_df)} measurements")
    if len(merged_df) == 0:
        print("No matching data found after merge. Checking image names...")
        print(f"Position data image names: {position_df['image_name'].unique()[:5]}")
        print(f"Measurement data image names: {measurement_df['image_name'].unique()[:5]}")
        return None
    
    # Create separate heatmaps for each pond type and size category
    pond_types = merged_df['pond_type'].unique()
    size_categories = merged_df['manual_size'].unique()
    
    print(f"\nCreating separate heatmaps for:")
    print(f"Pond types: {pond_types}")
    print(f"Size categories: {size_categories}")
    
    # Error metrics to analyze - focus on pixel to mm conversion errors
    error_metrics = [
        'pixel_difference_pct', 'scale_impact_ratio_rho', 'measurement_difference_pct'
    ]
    
    error_titles = [
        'Pixel Difference (%)', 'Scale Impact Ratio (ρ)', 'Measurement Difference (%)'
    ]
    
    # Create separate heatmaps for each combination
    for pond_type in pond_types:
        for size_category in size_categories:
            # Filter data for this combination
            subset_df = merged_df[
                (merged_df['pond_type'] == pond_type) & 
                (merged_df['manual_size'] == size_category)
            ]
            
            if len(subset_df) == 0:
                print(f"No data for {pond_type}-{size_category}")
                continue
            
            print(f"\nCreating heatmap for {pond_type}-{size_category} ({len(subset_df)} measurements)")
            
            # Create subplots for this combination
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=error_titles,
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Add scatter plots for each error metric
            for i, (error_metric, title) in enumerate(zip(error_metrics, error_titles)):
                row = 1
                col = i + 1
                
                # Create scatter plot
                scatter = go.Scatter(
                    x=subset_df['center_x'],
                    y=subset_df['center_y'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=subset_df[error_metric],
                        colorscale='RdBu_r',  # Red-Blue diverging colormap for errors
                        showscale=True,
                        colorbar=dict(title=title, len=0.8),
                        opacity=0.8
                    ),
                    text=[f"Image: {img}<br>Error: {err:.3f}<br>Position: ({x:.3f}, {y:.3f})" 
                          for img, err, x, y in zip(subset_df['image_name'], subset_df[error_metric], 
                                                   subset_df['center_x'], subset_df['center_y'])],
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    name=title
                )
                
                fig.add_trace(scatter, row=row, col=col)
                
                # Update axes labels
                fig.update_xaxes(title_text="X Position (normalized)", row=row, col=col, range=[0, 1])
                fig.update_yaxes(title_text="Y Position (normalized)", row=row, col=col, range=[0, 1])
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f'Measurement Error Heatmap: {pond_type} Pond - {size_category.capitalize()} Prawns',
                    'y':0.98,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 18}
                },
                height=500,
                width=1200,
                showlegend=False
            )
            
            # Save the plot
            filename = f"position_error_heatmap_{pond_type}_{size_category}.html"
            fig.write_html(filename)
            print(f"Saved {filename}")
            
            # Create individual summary plot for this combination
            summary_fig = go.Figure()
            
            # Use Scale Impact Ratio (ρ) as the primary pixel-to-mm conversion metric
            scatter = go.Scatter(
                x=subset_df['center_x'],
                y=subset_df['center_y'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=subset_df['scale_impact_ratio_rho'],
                    colorscale='RdBu_r',  # Red-Blue diverging colormap
                    showscale=True,
                    colorbar=dict(title="Scale Impact Ratio (ρ)"),
                    opacity=0.8
                ),
                text=[f"Image: {img}<br>Scale Ratio (ρ): {rho:.3f}<br>Position: ({x:.3f}, {y:.3f})<br>Pixel Diff: {pix_diff:.2f}%<br>Meas Diff: {mdiff:.2f}%" 
                      for img, rho, x, y, pix_diff, mdiff in zip(subset_df['image_name'], subset_df['scale_impact_ratio_rho'], 
                                               subset_df['center_x'], subset_df['center_y'],
                                               subset_df['pixel_difference_pct'], subset_df['measurement_difference_pct'])],
                hovertemplate='<b>%{text}</b><extra></extra>'
            )
            
            summary_fig.add_trace(scatter)
            
            # Add image center reference
            summary_fig.add_trace(go.Scatter(
                x=[0.5], y=[0.5],
                mode='markers',
                marker=dict(symbol='x', size=20, color='black'),
                name='Image Center',
                showlegend=True
            ))
            
            summary_fig.update_layout(
                title=f'Pixel-to-MM Conversion (Scale Ratio ρ): {pond_type} Pond - {size_category.capitalize()} Prawns',
                xaxis_title="X Position (normalized)",
                yaxis_title="Y Position (normalized)",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                height=700,
                width=900
            )
            
            summary_filename = f"position_error_summary_{pond_type}_{size_category}.html"
            summary_fig.write_html(summary_filename)
            print(f"Saved {summary_filename}")
            
            # Calculate and print statistics for this combination
            print(f"\n=== STATISTICS FOR {pond_type.upper()} - {size_category.upper()} ===")
            
            # Define image regions for this subset
            subset_df['image_region'] = 'center'
            subset_df.loc[subset_df['center_x'] < 0.33, 'image_region'] = 'left'
            subset_df.loc[subset_df['center_x'] > 0.67, 'image_region'] = 'right'
            subset_df.loc[subset_df['center_y'] < 0.33, 'image_region'] = 'top'
            subset_df.loc[subset_df['center_y'] > 0.67, 'image_region'] = 'bottom'
            
            region_stats = subset_df.groupby('image_region')[error_metrics].agg(['mean', 'std', 'count'])
            print(region_stats)
    
    # Create overall summary heatmap (all data combined)
    print("\n=== CREATING OVERALL SUMMARY HEATMAP ===")
    
    overall_fig = go.Figure()
    
    # Use different colors for each pond-size combination
    colors = ['red', 'blue', 'green', 'orange']
    color_idx = 0
    
    for pond_type in pond_types:
        for size_category in size_categories:
            subset_df = merged_df[
                (merged_df['pond_type'] == pond_type) & 
                (merged_df['manual_size'] == size_category)
            ]
            
            if len(subset_df) == 0:
                continue
            
            scatter = go.Scatter(
                x=subset_df['center_x'],
                y=subset_df['center_y'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=subset_df['scale_impact_ratio_rho'],
                    colorscale='RdBu_r',
                    showscale=True,
                    colorbar=dict(title="Scale Impact Ratio (ρ)"),
                    opacity=0.7
                ),
                text=[f"Image: {img}<br>Scale Ratio (ρ): {rho:.3f}<br>Position: ({x:.3f}, {y:.3f})<br>Pond: {pond}<br>Size: {size}" 
                      for img, rho, x, y, pond, size in zip(subset_df['image_name'], subset_df['scale_impact_ratio_rho'], 
                                                           subset_df['center_x'], subset_df['center_y'],
                                                           subset_df['pond_type'], subset_df['manual_size'])],
                hovertemplate='<b>%{text}</b><extra></extra>',
                name=f"{pond_type}-{size_category}"
            )
            
            overall_fig.add_trace(scatter)
    
    # Add image center reference
    overall_fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5],
        mode='markers',
        marker=dict(symbol='x', size=20, color='black'),
        name='Image Center',
        showlegend=True
    ))
    
    overall_fig.update_layout(
        title='Pixel-to-MM Conversion (Scale Ratio ρ) Distribution by Position and Category',
        xaxis_title="X Position (normalized)",
        yaxis_title="Y Position (normalized)",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=800,
        width=1000
    )
    
    overall_fig.write_html("position_error_overall_summary.html")
    print("Saved position_error_overall_summary.html")
    
    # Save the complete data
    merged_df.to_csv("position_error_heatmap_data.csv", index=False)
    print("Saved data to: position_error_heatmap_data.csv")
    
    # Print overall statistics
    print("\n=== OVERALL STATISTICS ===")
    print(f"Total measurements: {len(merged_df)}")
    for pond_type in pond_types:
        for size_category in size_categories:
            subset = merged_df[
                (merged_df['pond_type'] == pond_type) & 
                (merged_df['manual_size'] == size_category)
            ]
            if len(subset) > 0:
                print(f"{pond_type}-{size_category}: {len(subset)} measurements, MAE mean: {subset['individual_mae'].mean():.2f}mm")
    
    return merged_df

if __name__ == "__main__":
    data = create_position_error_heatmap() 