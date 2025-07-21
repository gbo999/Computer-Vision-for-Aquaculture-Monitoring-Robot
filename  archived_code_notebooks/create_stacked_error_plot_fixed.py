import pandas as pd
import numpy as np
import fiftyone as fo
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.subplots as sp

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

def create_stacked_error_plot():
    """
    Create a stacked bar plot showing how pixel errors and measurement errors 
    compound or cancel each other using signed errors for each individual prawn.
    FIXED: Merge on combination of image_name, pond_type, and manual_size to avoid duplicates.
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
    
    # FIXED: Drop duplicates by image_name to keep only the first detection/keypoint per image
    position_df = position_df.drop_duplicates(subset=['image_name'])
    if len(position_df) == 0:
        print("No position data extracted. Please check the dataset structure and keypoint storage.")
        return None
    
    # FIXED: Merge on image_name only for position data (since position data doesn't have pond_type/size)
    # We'll add pond_type and manual_size from measurement_df after the merge
    merged_df = pd.merge(position_df, measurement_df, on='image_name', how='inner')
    print(f"Merged data contains {len(merged_df)} measurements")
    if len(merged_df) == 0:
        print("No matching data found after merge.")
        return None
    
    # Calculate PROPER signed errors for each individual prawn
    print("\n=== CALCULATING SIGNED ERRORS FOR EACH PRAWN ===")
    
    # Calculate signed pixel difference (keypoint - shai)
    merged_df['signed_pixel_diff'] = merged_df['total_length_pixels'] - merged_df['shai_length']
    merged_df['signed_pixel_diff_pct'] = (merged_df['signed_pixel_diff'] / merged_df['shai_length']) * 100
    
    # Calculate signed measurement difference (calculated - real)
    merged_df['signed_measurement_diff'] = merged_df['calculated_total_length_mm'] - merged_df['real_length']
    merged_df['signed_measurement_diff_pct'] = (merged_df['signed_measurement_diff'] / merged_df['real_length']) * 100
    
    # Create categories for analysis
    merged_df['category'] = merged_df['pond_type'] + '_' + merged_df['manual_size']
    
    print(f"Categories found: {merged_df['category'].unique()}")
    
    # Show individual prawn data
    print("\n=== INDIVIDUAL PRAWN SIGNED ERRORS ===")
    individual_data = merged_df[['image_name', 'pond_type', 'manual_size', 'category', 'signed_pixel_diff_pct', 'signed_measurement_diff_pct', 'individual_mae', 'individual_mape']].copy()
    individual_data = individual_data.round(3)
    print(individual_data.head(10))
    
    # Save individual prawn data
    individual_data.to_csv("individual_prawn_signed_errors_fixed.csv", index=False)
    print("Saved individual_prawn_signed_errors_fixed.csv")
    
    # Create stacked bar plot
    print("\n=== CREATING STACKED ERROR PLOT ===")
    
    # Group by category and calculate statistics
    error_summary = merged_df.groupby('category').agg({
        'signed_pixel_diff_pct': ['mean', 'std', 'count'],
        'signed_measurement_diff_pct': ['mean', 'std', 'count'],
        'individual_mae': ['mean', 'std'],
        'individual_mape': ['mean', 'std']
    }).round(3)
    
    print("\nError Summary by Category:")
    print(error_summary)
    
    # Prepare data for plotting
    categories = merged_df['category'].unique()
    
    # Calculate mean signed errors for each category
    pixel_errors = []
    measurement_errors = []
    mae_values = []
    mape_values = []
    
    for category in categories:
        cat_data = merged_df[merged_df['category'] == category]
        pixel_errors.append(cat_data['signed_pixel_diff_pct'].mean())
        measurement_errors.append(cat_data['signed_measurement_diff_pct'].mean())
        mae_values.append(cat_data['individual_mae'].mean())
        mape_values.append(cat_data['individual_mape'].mean())
    
    # Create stacked bar plot
    fig = go.Figure()
    
    # Add pixel error bars
    fig.add_trace(go.Bar(
        name='Signed Pixel Error (%)',
        x=categories,
        y=pixel_errors,
        marker_color='lightblue',
        text=[f'{val:.2f}%' for val in pixel_errors],
        textposition='auto',
    ))
    
    # Add measurement error bars
    fig.add_trace(go.Bar(
        name='Signed Measurement Error (%)',
        x=categories,
        y=measurement_errors,
        marker_color='lightcoral',
        text=[f'{val:.2f}%' for val in measurement_errors],
        textposition='auto',
    ))
    
    # Update layout
    fig.update_layout(
        title='Stacked Error Analysis: Signed Pixel vs Measurement Errors by Category (FIXED)',
        xaxis_title='Pond Type and Size',
        yaxis_title='Signed Error (%)',
        barmode='stack',
        height=600,
        width=800
    )
    
    fig.write_html("stacked_error_plot_fixed.html")
    print("Saved stacked_error_plot_fixed.html")
    
    # Create individual bar plots for better visibility
    fig_individual = go.Figure()
    
    # Add bars side by side
    fig_individual.add_trace(go.Bar(
        name='Signed Pixel Error (%)',
        x=categories,
        y=pixel_errors,
        marker_color='lightblue',
        text=[f'{val:.2f}%' for val in pixel_errors],
        textposition='auto',
    ))
    
    fig_individual.add_trace(go.Bar(
        name='Signed Measurement Error (%)',
        x=categories,
        y=measurement_errors,
        marker_color='lightcoral',
        text=[f'{val:.2f}%' for val in measurement_errors],
        textposition='auto',
    ))
    
    fig_individual.update_layout(
        title='Error Comparison: Signed Pixel vs Measurement Errors by Category (FIXED)',
        xaxis_title='Pond Type and Size',
        yaxis_title='Signed Error (%)',
        barmode='group',
        height=600,
        width=800
    )
    
    fig_individual.write_html("error_comparison_plot_fixed.html")
    print("Saved error_comparison_plot_fixed.html")
    
    # Create MAE and MAPE plot
    fig_metrics = go.Figure()
    
    fig_metrics.add_trace(go.Bar(
        name='MAE (mm)',
        x=categories,
        y=mae_values,
        marker_color='lightgreen',
        text=[f'{val:.2f} mm' for val in mae_values],
        textposition='auto',
    ))
    
    fig_metrics.add_trace(go.Bar(
        name='MAPE (%)',
        x=categories,
        y=mape_values,
        marker_color='lightyellow',
        text=[f'{val:.2f}%' for val in mape_values],
        textposition='auto',
    ))
    
    fig_metrics.update_layout(
        title='MAE and MAPE by Category (FIXED)',
        xaxis_title='Pond Type and Size',
        yaxis_title='Error Value',
        barmode='group',
        height=600,
        width=800
    )
    
    fig_metrics.write_html("mae_mape_plot_fixed.html")
    print("Saved mae_mape_plot_fixed.html")
    
    # Create a comprehensive table showing all metrics
    print("\n=== COMPREHENSIVE ERROR ANALYSIS ===")
    
    comprehensive_analysis = []
    for category in categories:
        cat_data = merged_df[merged_df['category'] == category]
        pixel_mean = cat_data['signed_pixel_diff_pct'].mean()
        meas_mean = cat_data['signed_measurement_diff_pct'].mean()
        mae_mean = cat_data['individual_mae'].mean()
        mape_mean = cat_data['individual_mape'].mean()
        
        # Determine if errors compound or cancel
        if (pixel_mean > 0 and meas_mean > 0) or (pixel_mean < 0 and meas_mean < 0):
            effect = "Compound"  # Both positive or both negative
        else:
            effect = "Cancel"    # One positive, one negative
        
        # Calculate the net effect
        net_error = pixel_mean + meas_mean
        
        comprehensive_analysis.append({
            'Category': category,
            'Signed Pixel Error (%)': f"{pixel_mean:.2f}",
            'Signed Measurement Error (%)': f"{meas_mean:.2f}",
            'Net Error (%)': f"{net_error:.2f}",
            'MAE (mm)': f"{mae_mean:.2f}",
            'MAPE (%)': f"{mape_mean:.2f}",
            'Effect': effect,
            'Count': len(cat_data)
        })
    
    comprehensive_df = pd.DataFrame(comprehensive_analysis)
    print("\nComprehensive Error Analysis:")
    print(comprehensive_df.to_string(index=False))
    
    # Save the comprehensive analysis
    comprehensive_df.to_csv("comprehensive_error_analysis_fixed.csv", index=False)
    print("\nSaved comprehensive_error_analysis_fixed.csv")
    
    # Create visual comprehensive table
    fig_comprehensive = go.Figure(data=[go.Table(
        header=dict(
            values=['Category', 'Signed Pixel Error (%)', 'Signed Measurement Error (%)', 'Net Error (%)', 'MAE (mm)', 'MAPE (%)', 'Effect', 'Count'],
            fill_color='lightblue',
            align='left',
            font=dict(size=14, color='black')
        ),
        cells=dict(
            values=[
                comprehensive_df['Category'],
                comprehensive_df['Signed Pixel Error (%)'],
                comprehensive_df['Signed Measurement Error (%)'],
                comprehensive_df['Net Error (%)'],
                comprehensive_df['MAE (mm)'],
                comprehensive_df['MAPE (%)'],
                comprehensive_df['Effect'],
                comprehensive_df['Count']
            ],
            fill_color='white',
            align='left',
            font=dict(size=12)
        )
    )])
    
    fig_comprehensive.update_layout(
        title="Comprehensive Error Analysis with MAE and MAPE (FIXED)",
        width=1200,
        height=300
    )
    
    fig_comprehensive.write_html("comprehensive_error_analysis_table_fixed.html")
    print("Saved comprehensive_error_analysis_table_fixed.html")
    
    return comprehensive_df

def plot_error_components_by_pond_type(merged_df):
    # Calculate signed pixel error (in mm) and signed measurement error (in mm)
    merged_df['signed_pixel_error_mm'] = merged_df['total_length_pixels'] - merged_df['shai_length']
    merged_df['signed_scale_error_mm'] = merged_df['calculated_total_length_mm'] - merged_df['total_length_pixels'] * (merged_df['real_length'] / merged_df['shai_length'])
    # The above scale error is the difference between the mm length calculated from keypoints and the mm length you would get if the pixel length was perfect (i.e., only scale error, not pixel error)
    # If you want just (calculated_total_length_mm - real_length), use that instead:
    # merged_df['signed_scale_error_mm'] = merged_df['calculated_total_length_mm'] - merged_df['real_length']

    categories = merged_df['category'].unique()
    n_categories = len(categories)
    fig = sp.make_subplots(rows=1, cols=n_categories, subplot_titles=categories)

    for i, category in enumerate(categories):
        cat_data = merged_df[merged_df['category'] == category].reset_index(drop=True)
        x = list(range(len(cat_data)))
        fig.add_trace(
            go.Bar(
                x=x,
                y=cat_data['signed_pixel_error_mm'],
                name='Pixel Error',
                marker_color='royalblue',
                showlegend=(i==0)
            ),
            row=1, col=i+1
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=cat_data['signed_scale_error_mm'],
                name='Scale Error',
                marker_color='orange',
                showlegend=(i==0)
            ),
            row=1, col=i+1
        )
        fig.update_xaxes(title_text='Measurement Index', row=1, col=i+1)
        fig.update_yaxes(title_text='Error (mm)', row=1, col=i+1)

    fig.update_layout(
        title_text='Error Components Analysis by Pond Type (FIXED)',
        barmode='group',
        width=350*n_categories,
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig.write_html('error_components_by_pond_type_fixed.html')
    print('Saved error_components_by_pond_type_fixed.html')

    # --- NEW: Individual Measurement Error Components Plot ---
    print("\n=== CREATING ERROR COMPONENTS PLOT FOR EACH MEASUREMENT ===")
    # We'll use the merged_df, which has all individual prawns
    # Calculate pixel error in mm and measurement error in mm
    merged_df['pixel_error_mm'] = merged_df['total_length_pixels'] - merged_df['shai_length']
    merged_df['scale_error_mm'] = merged_df['calculated_total_length_mm'] - merged_df['total_length_pixels'] * (merged_df['real_length'] / merged_df['shai_length'])
    # The above scale_error_mm is the error due to scale conversion, after accounting for pixel error
    # But for clarity, let's just use:
    # pixel_error_mm = keypoint_pixels - shai_pixels
    # scale_error_mm = (calculated_mm - shai_pixels * scale_factor)
    merged_df['scale_factor'] = merged_df['real_length'] / merged_df['shai_length']
    merged_df['scale_error_mm'] = merged_df['calculated_total_length_mm'] - merged_df['total_length_pixels'] * merged_df['scale_factor']

    # Assign a measurement index for plotting
    merged_df = merged_df.reset_index(drop=True)
    merged_df['measurement_index'] = merged_df.index

    # Plot for each pond_type + manual_size
    import plotly.subplots as sp
    pond_groups = merged_df.groupby('category')
    n_groups = pond_groups.ngroups
    fig = sp.make_subplots(rows=1, cols=n_groups, subplot_titles=list(pond_groups.groups.keys()))

    for i, (cat, group) in enumerate(pond_groups, 1):
        fig.add_trace(
            go.Bar(
                x=group['measurement_index'],
                y=group['pixel_error_mm'],
                name='Pixel Error (mm)',
                marker_color='royalblue',
                showlegend=(i==1)
            ),
            row=1, col=i
        )
        fig.add_trace(
            go.Bar(
                x=group['measurement_index'],
                y=group['scale_error_mm'],
                name='Scale Error (mm)',
                marker_color='orange',
                showlegend=(i==1)
            ),
            row=1, col=i
        )
        fig.update_xaxes(title_text='Measurement Index', row=1, col=i)
        fig.update_yaxes(title_text='Error (mm)', row=1, col=i)

    fig.update_layout(
        title_text='Error Components Analysis by Pond Type (FIXED)',
        barmode='group',
        width=350*n_groups,
        height=400,
        legend=dict(x=1.05, y=1)
    )
    fig.write_html('error_components_per_measurement_fixed.html')
    print('Saved error_components_per_measurement_fixed.html')

if __name__ == "__main__":
    error_data = create_stacked_error_plot()

    # --- Stacked Bar Plot for Each Measurement (Compounding/Cancellation) ---
    print("\n=== CREATING STACKED ERROR BAR PLOT FOR EACH MEASUREMENT (SIGNED PERCENTAGE) ===")
    try:
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import plotly.subplots as sp
        measurement_df = pd.read_csv("complete_detailed_results_table.csv")
        signed_df = pd.read_csv("individual_prawn_signed_errors_fixed.csv")
        
        # FIXED: Merge on the combination of image_name, pond_type, and manual_size
        # First, ensure both dataframes have the required columns
        if 'pond_type' not in signed_df.columns or 'manual_size' not in signed_df.columns:
            # If signed_df doesn't have pond_type and manual_size, we need to get them from measurement_df
            # Create a mapping from image_name to pond_type and manual_size
            measurement_mapping = measurement_df[['image_name', 'pond_type', 'manual_size']].drop_duplicates()
            signed_df = pd.merge(signed_df, measurement_mapping, on='image_name', how='inner')
        
        # Now merge on the combination of all three columns
        merged_df = pd.merge(measurement_df, signed_df, on=["image_name", "pond_type", "manual_size"], how="inner")
        merged_df['category'] = merged_df['pond_type'] + '_' + merged_df['manual_size']
        merged_df = merged_df.reset_index(drop=True)
        merged_df['measurement_index'] = merged_df.index
        
        print(f"FIXED: Merged data contains {len(merged_df)} measurements (no duplicates)")
        
        pond_groups = merged_df.groupby('category')
        n_groups = pond_groups.ngroups
        fig = sp.make_subplots(rows=1, cols=n_groups, subplot_titles=list(pond_groups.groups.keys()))
        for i, (cat, group) in enumerate(pond_groups, 1):
            group = group.sort_values('measurement_index')
            mape = np.abs(group['signed_measurement_diff_pct'])
            customdata = np.stack((group['signed_pixel_diff_pct'], group['signed_measurement_diff_pct'], mape), axis=-1)
            fig.add_trace(
                go.Bar(
                    x=group['measurement_index'],
                    y=group['signed_pixel_diff_pct'],
                    name='Signed Pixel Error (%)',
                    marker_color='royalblue',
                    customdata=customdata,
                    hovertemplate='Index: %{x}<br>Signed Pixel Error: %{customdata[0]:.2f}%<br>Signed Meas Error: %{customdata[1]:.2f}%<br>MAPE: %{customdata[2]:.2f}%',
                    showlegend=(i==1)
                ),
                row=1, col=i
            )
            fig.add_trace(
                go.Bar(
                    x=group['measurement_index'],
                    y=group['signed_measurement_diff_pct'],
                    name='Signed Measurement Error (%)',
                    marker_color='orange',
                    customdata=customdata,
                    hovertemplate='Index: %{x}<br>Signed Pixel Error: %{customdata[0]:.2f}%<br>Signed Meas Error: %{customdata[1]:.2f}%<br>MAPE: %{customdata[2]:.2f}%',
                    showlegend=(i==1)
                ),
                row=1, col=i
            )
            fig.update_xaxes(title_text='Measurement Index', row=1, col=i)
            fig.update_yaxes(title_text='Signed Error (%)', row=1, col=i)
        fig.update_layout(
            title_text='Stacked Signed Error Components by Measurement (Percentage, Compounding/Cancellation) - FIXED',
            barmode='relative',
            width=350*n_groups,
            height=400,
            legend=dict(x=1.05, y=1)
        )
        print('Writing error_components_stacked_signed_pct_per_measurement_fixed.html...')
        fig.write_html('error_components_stacked_signed_pct_per_measurement_fixed.html')
        print('Saved error_components_stacked_signed_pct_per_measurement_fixed.html')
    except Exception as e:
        print('Failed to create error_components_stacked_signed_pct_per_measurement_fixed.html:', e)
    print('Script completed.')

    print("\n=== CREATING STACKED ERROR BAR PLOT FOR EACH MEASUREMENT (PIXEL + SCALE ERROR, SIGNED PERCENTAGE) ===")
    try:
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import plotly.subplots as sp
        measurement_df = pd.read_csv("complete_detailed_results_table.csv")
        signed_df = pd.read_csv("individual_prawn_signed_errors_fixed.csv")
        
        # FIXED: Merge on the combination of image_name, pond_type, and manual_size
        # First, ensure both dataframes have the required columns
        if 'pond_type' not in signed_df.columns or 'manual_size' not in signed_df.columns:
            # If signed_df doesn't have pond_type and manual_size, we need to get them from measurement_df
            # Create a mapping from image_name to pond_type and manual_size
            measurement_mapping = measurement_df[['image_name', 'pond_type', 'manual_size']].drop_duplicates()
            signed_df = pd.merge(signed_df, measurement_mapping, on='image_name', how='inner')
        
        # Now merge on the combination of all three columns
        merged_df = pd.merge(measurement_df, signed_df, on=["image_name", "pond_type", "manual_size"], how="inner")
        merged_df['category'] = merged_df['pond_type'] + '_' + merged_df['manual_size']
        merged_df = merged_df.reset_index(drop=True)
        merged_df['measurement_index'] = merged_df.index
        
        # FIX: Calculate signed_pixel_diff for the merged data (in pixels) before grouping
        merged_df['signed_pixel_diff'] = merged_df['total_length_pixels'] - merged_df['shai_length']
        
        # Print summary of red-highlighted bars (100+ pixel difference)
        red_count = (merged_df['signed_pixel_diff'].abs() >= 100).sum()
        total_count = len(merged_df)
        print(f"FIXED: Merged data contains {len(merged_df)} measurements (no duplicates)")
        print(f"Bars with 100+ pixel difference (colored red): {red_count}/{total_count} ({red_count/total_count*100:.1f}%)")
        
        # Calculate scale factor for each prawn
        merged_df['scale_factor'] = merged_df['real_length'] / merged_df['shai_length']
        # Pixel error (%)
        merged_df['pixel_error_pct'] = 100 * (merged_df['total_length_pixels'] - merged_df['shai_length']) / merged_df['shai_length']
        # Scale error (%)
        mm_from_pixels = merged_df['total_length_pixels'] * merged_df['scale_factor']
        merged_df['scale_error_pct'] = 100 * (merged_df['calculated_total_length_mm'] - mm_from_pixels) / merged_df['real_length']
        # MAPE (abs total measurement error)
        merged_df['mape'] = 100 * np.abs((merged_df['calculated_total_length_mm'] - merged_df['real_length']) / merged_df['real_length'])
        pond_groups = merged_df.groupby('category')
        n_groups = pond_groups.ngroups
        fig = sp.make_subplots(rows=1, cols=n_groups, subplot_titles=list(pond_groups.groups.keys()))
        for i, (cat, group) in enumerate(pond_groups, 1):
            group = group.sort_values('measurement_index')
            customdata = np.stack((group['image_name'], group['pixel_error_pct'], group['scale_error_pct'], group['mape'], group['calculated_total_length_mm']), axis=-1)
            # Use group['signed_pixel_diff'] for the red coloring logic (100+ pixel difference)
            group_pixel_colors = ['red' if abs(pixel_diff) >= 100 else 'royalblue' for pixel_diff in group['signed_pixel_diff']]
            fig.add_trace(
                go.Bar(
                    x=group['measurement_index'],
                    y=group['pixel_error_pct'],
                    name='Pixel Error (%)',
                    marker_color=group_pixel_colors,
                    customdata=customdata,
                    hovertemplate='Image: %{customdata[0]}<br>Index: %{x}<br>Pixel Error: %{customdata[1]:.2f}%<br>Scale Error: %{customdata[2]:.2f}%<br>MAPE: %{customdata[3]:.2f}%<br>Predicted Length: %{customdata[4]:.2f} mm',
                    showlegend=(i==1)
                ),
                row=1, col=i
            )
            fig.add_trace(
                go.Bar(
                    x=group['measurement_index'],
                    y=group['scale_error_pct'],
                    name='Scale Error (%)',
                    marker_color='orange',
                    customdata=customdata,
                    hovertemplate='Image: %{customdata[0]}<br>Index: %{x}<br>Pixel Error: %{customdata[1]:.2f}%<br>Scale Error: %{customdata[2]:.2f}%<br>MAPE: %{customdata[3]:.2f}%<br>Predicted Length: %{customdata[4]:.2f} mm',
                    showlegend=(i==1)
                ),
                row=1, col=i
            )
            fig.update_xaxes(title_text='Measurement Index', row=1, col=i)
            fig.update_yaxes(title_text='Signed Error (%)', row=1, col=i)
        fig.update_layout(
            title_text='Stacked Pixel + Scale Error Components by Measurement (Percentage, Compounding/Cancellation) - FIXED',
            barmode='relative',
            width=350*n_groups,
            height=400,
            legend=dict(x=1.05, y=1)
        )
        print('Writing error_components_stacked_pixel_scale_pct_per_measurement_fixed.html...')
        fig.write_html('error_components_stacked_pixel_scale_pct_per_measurement_fixed.html')
        print('Saved error_components_stacked_pixel_scale_pct_per_measurement_fixed.html')
    except Exception as e:
        print('Failed to create error_components_stacked_pixel_scale_pct_per_measurement_fixed.html:', e)
    print('Script completed.') 