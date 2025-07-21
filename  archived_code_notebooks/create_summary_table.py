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
    
    # Calculate pixel error (%) and scale error (%)
    merged_df['scale_factor'] = merged_df['real_length'] / merged_df['shai_length']
    merged_df['pixel_error_pct'] = 100 * (merged_df['total_length_pixels'] - merged_df['shai_length']) / merged_df['shai_length']
    mm_from_pixels = merged_df['total_length_pixels'] * merged_df['scale_factor']
    merged_df['scale_error_pct'] = 100 * (merged_df['calculated_total_length_mm'] - mm_from_pixels) / merged_df['real_length']
    # Group by pond_type and manual_size
    summary = merged_df.groupby(['pond_type', 'manual_size']).agg(
        mae_mm = ('individual_mae', 'mean'),
        mape_pct = ('individual_mape', 'mean'),
        pixel_error_pct_mean = ('pixel_error_pct', 'mean'),
        pixel_error_pct_std = ('pixel_error_pct', 'std'),
        scale_error_pct_mean = ('scale_error_pct', 'mean'),
        scale_error_pct_std = ('scale_error_pct', 'std'),
        count = ('image_name', 'count')
    ).reset_index()
    print("\nSummary Table (Pond, Size):\n", summary)
    summary.to_csv("summary_table_pixel_scale_error.csv", index=False)
    with open("summary_table_pixel_scale_error.txt", "w") as f:
        f.write(summary.to_string(index=False))
    print("\nSaved summary_table_pixel_scale_error.csv and summary_table_pixel_scale_error.txt")
    return summary

if __name__ == "__main__":
    create_summary_table() 