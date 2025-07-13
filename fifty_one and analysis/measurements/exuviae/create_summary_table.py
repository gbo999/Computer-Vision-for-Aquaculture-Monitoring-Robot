import pandas as pd
import numpy as np
import fiftyone as fo
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

def create_summary_table():
    """
    Create a summary table with MAE, MAPE, pixel percentage, and measurement percentage 
    for each pond type and size category.
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
        print("No matching data found after merge.")
        return None
    
    # Create summary table by pond type and size
    print("\n=== SUMMARY TABLE BY POND TYPE AND SIZE ===")
    
    # Define the metrics to summarize
    metrics = ['individual_mae', 'individual_mape', 'pixel_difference_pct', 'measurement_difference_pct']
    metric_names = ['MAE (mm)', 'MAPE (%)', 'Pixel Difference (%)', 'Measurement Difference (%)']
    
    # Create summary statistics
    summary_stats = merged_df.groupby(['pond_type', 'manual_size'])[metrics].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    
    print("\nDetailed Summary Statistics:")
    print(summary_stats)
    
    # Create a cleaner summary table
    clean_summary = merged_df.groupby(['pond_type', 'manual_size'])[metrics].agg([
        'count', 'mean', 'std'
    ]).round(3)
    
    # Flatten column names
    clean_summary.columns = [f"{col[1]}_{col[0]}" for col in clean_summary.columns]
    
    print("\nClean Summary Table:")
    print(clean_summary)
    
    # Create a formatted table for display
    formatted_summary = []
    
    for (pond_type, size), group_data in merged_df.groupby(['pond_type', 'manual_size']):
        row = {
            'Pond Type': pond_type,
            'Size': size,
            'Count': len(group_data),
            'MAE (mm)': f"{group_data['individual_mae'].mean():.2f} ± {group_data['individual_mae'].std():.2f}",
            'MAPE (%)': f"{group_data['individual_mape'].mean():.2f} ± {group_data['individual_mape'].std():.2f}",
            'Pixel Diff (%)': f"{group_data['pixel_difference_pct'].mean():.2f} ± {group_data['pixel_difference_pct'].std():.2f}",
            'Meas Diff (%)': f"{group_data['measurement_difference_pct'].mean():.2f} ± {group_data['measurement_difference_pct'].std():.2f}"
        }
        formatted_summary.append(row)
    
    formatted_df = pd.DataFrame(formatted_summary)
    print("\nFormatted Summary Table (Mean ± Std):")
    print(formatted_df.to_string(index=False))
    
    # Save the summary tables
    summary_stats.to_csv("detailed_summary_statistics.csv")
    clean_summary.to_csv("clean_summary_table.csv")
    formatted_df.to_csv("formatted_summary_table.csv")
    
    print("\nFiles saved:")
    print("- detailed_summary_statistics.csv: Full statistics (count, mean, std, min, max)")
    print("- clean_summary_table.csv: Clean format with flattened column names")
    print("- formatted_summary_table.csv: Formatted table with mean ± std")
    
    # Create a visual summary table using Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Pond Type', 'Size', 'Count', 'MAE (mm)', 'MAPE (%)', 'Pixel Diff (%)', 'Meas Diff (%)'],
            fill_color='lightblue',
            align='left',
            font=dict(size=14, color='black')
        ),
        cells=dict(
            values=[
                formatted_df['Pond Type'],
                formatted_df['Size'],
                formatted_df['Count'],
                formatted_df['MAE (mm)'],
                formatted_df['MAPE (%)'],
                formatted_df['Pixel Diff (%)'],
                formatted_df['Meas Diff (%)']
            ],
            fill_color='white',
            align='left',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        title="Summary Table: Error Metrics by Pond Type and Size",
        width=1000,
        height=300
    )
    
    fig.write_html("summary_table_visual.html")
    print("- summary_table_visual.html: Interactive visual table")
    
    return formatted_df

if __name__ == "__main__":
    summary_data = create_summary_table() 