import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import fiftyone as fo
import os
import ast
import plotly.express as px
from sklearn.metrics import r2_score
import warnings
from scipy import stats
from plotly.subplots import make_subplots

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# For specific matplotlib warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module="plotly")
# Suppress pandas SettingWithCopyWarning
warnings.filterwarnings("ignore", message=".*SettingWithCopyWarning.*")
# Additional method to suppress pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'

def calculate_mape(estimated_lengths, true_lengths):
    """
Calculate Mean Absolute Percentage Error between estimated and true lengths.
Parameters:
-----------
estimated_lengths : array-like
    Estimated length measurements (Len_e)
true_lengths : array-like
    True length measurements (Len_t)
Returns:
--------
list
    List of individual absolute percentage errors for each measurement pair
"""
# Calculate individual absolute percentage errors
# Formula: |estimated - true| / estimated * 100
    absolute_percentage_errors = [abs(est - true) / est * 100 for est, true in zip(estimated_lengths, true_lengths)]

    return absolute_percentage_errors




def main():
    parser = argparse.ArgumentParser('flags_analysis')
    parser.add_argument('--type', type=str,choices=['carapace','body'], default='carapace')
    parser.add_argument('--weights_type', type=str,choices=['all','kalkar','car'], default='all')
    parser.add_argument('--error_size', type=str,choices=['min','median','mean','max'], default='mean')
    args = parser.parse_args()




    """
Error Flags Analysis Script
--------------------------
This script analyzes measurement errors in prawn length data and identifies potential error sources
through a series of flags. It visualizes relationships between error flags, error rates, and measurement 
characteristics using both Matplotlib/Seaborn and Plotly.
Key components:
1. Data loading and preprocessing
2. Error calculation and flagging
3. Statistical analysis of errors and flags
4. Visualization of findings through various plot types

"""

   

# ----- Data Loading and Preprocessing -----

# Load the dataset from Excel file
    data_path = f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/updated_filtered_data_with_lengths_{args.type}-{args.weights_type}.xlsx'




    df = pd.read_excel(data_path)

# Remove any rows with missing values to ensure clean analysis
    df = df.dropna()

# Standardize pond type names for consistency in analysis and visualization
    df['Pond_Type'] = df['Pond_Type'].replace({
    'car': 'square',          # Rename 'car' to 'square'
    'right': 'circle_female', # Rename 'right' to 'circle_female'
    'left': 'circle_male',    # Rename 'left' to 'circle_male'
})
    #number of prawns in each pond type
    print(df['Pond_Type'].value_counts())



    df['std_length'] = df[['Length_1', 'Length_2', 'Length_3']].std(axis=1)


    print('========std length========')
    for pond_type in df['Pond_Type'].unique():
        print(f"Std length for {pond_type}: {df[df['Pond_Type'] == pond_type]['std_length'].mean()}")


    #mean length
    df['mean_length'] = df[['Length_1', 'Length_2', 'Length_3']].mean(axis=1)
    print('========mean length========')
    for pond_type in df['Pond_Type'].unique():
        print(f"Mean length for {pond_type}: {df[df['Pond_Type'] == pond_type]['mean_length'].describe()}")
        








    df['choice'] = f'{args.type}_{args.weights_type}_{args.error_size}'

    df['mean_scale'] = df[['Scale_1', 'Scale_2', 'Scale_3']].mean(axis=1)
    print('========mean scale========')
    print(df['mean_scale'])

    df['pred_scale'] = df['pred_Distance_pixels'] / df['Length_fov(mm)'] * 10
    print('========pred scale========')
    print(df['pred_scale'])

    df['Length_fov(mm)'] = df['Length_fov(mm)']




    # #uncertainty std / sqrt(3) 
    # df['uncertainty'] = df['Std_Length'] / np.sqrt(3)
    # print('========uncertainty========')
    # #describe uncertainty
    # print(df['uncertainty'].describe())


    #if pose <0.75 remove
    df = df[df['pose_eval_iou'] >= 0.75]


    #lngth fov bigger than 5
    df = df[df['Length_fov(mm)'] >= 5]


    #for each row the is 3 lengths need check all exist, if not exist, need to remove
    df = df[df['Length_1'].notna()]
    df = df[df['Length_2'].notna()]
    df = df[df['Length_3'].notna()]


    #std of length
    df['std_length'] = df[['Length_1', 'Length_2', 'Length_3']].std(axis=1)

    #describe std length
    print('========std length========')
    print(df['std_length'].describe())





    # #standard error of the mean
    # df['sem_length'] = df['std_length'] / np.sqrt(3)

    # #describe sem length
    # print('========sem length========')
    # print(df['sem_length'].describe())
    
    

# ----- Error Calculation -----
    df['annotation_length_1'] = df['Length_ground_truth_annotation_pixels'] / df['Scale_1'] * 10
    df['annotation_length_2'] = df['Length_ground_truth_annotation_pixels'] / df['Scale_2'] * 10
    df['annotation_length_3'] = df['Length_ground_truth_annotation_pixels'] / df['Scale_3'] * 10
    df['MPE_length1'] = abs(df['Length_1'] - df['Length_fov(mm)']) / df['Length_1'] * 100
    df['MPE_length2'] = abs(df['Length_2'] - df['Length_fov(mm)']) / df['Length_2'] * 100
    df['MPE_length3'] = abs(df['Length_3'] - df['Length_fov(mm)']) / df['Length_3'] * 100



    #mean annotation length
    df['mean_annotation_length'] = df[['annotation_length_1', 'annotation_length_2', 'annotation_length_3']].mean(axis=1)

    df['mean_length'] = df[['Length_1', 'Length_2', 'Length_3']].mean(axis=1)

    df['std_length'] = df[['Length_1', 'Length_2', 'Length_3']].std(axis=1)


   





    mean = df['mean_length'].mean()

    print('========mean========')
    print(mean)

    print('========std========')
    std = df['std_length'].mean()
    print(std)

    df['mean_pixels'] = df[['Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].mean(axis=1)


    df['diff_pixels'] = abs(df['mean_pixels'] - df['pred_Distance_pixels'])
    df['diff_mm'] =abs( df['Length_fov(mm)'] - df['mean_length'])

    df['mean_scale'] = df[['Scale_1', 'Scale_2', 'Scale_3']].mean(axis=1)

    df['pred_scale'] = ((df['pred_Distance_pixels'] / df['Length_fov(mm)']))*10

    df['diff_scale'] = abs(df['mean_scale'] - df['pred_scale'])

    

    # Find outliers using mean ± 3 standard deviations
    df_outliers = df[(df['Length_fov(mm)'] >= df['mean_length'] + 3*std) | 
            (df['Length_fov(mm)'] <= df['mean_length'] - 3*std)]



    # Print outliers before removing them
    for pond_type in df_outliers['Pond_Type'].unique():
        #print mean length and std length
       
        #print number of prawns in pond_type
        print(f"Number of prawns in {pond_type}: {len(df[df['Pond_Type'] == pond_type])}")
        #if pond_type is not circle_male or circle_female, remove   
        pond_outliers = df_outliers[df_outliers['Pond_Type'] == pond_type]
        print(f"\nOutliers being removed for {pond_type}:")
        print(pond_outliers[['Label', 'PrawnID', 'Length_fov(mm)', 'mean_length','diff_mm','pred_Distance_pixels','mean_pixels','diff_pixels','mean_scale','pred_scale','diff_scale']].sort_values(by='diff_mm', ascending=False))
        print(f"Number of outliers for {pond_type}: {len(pond_outliers)} out of {len(df[df['Pond_Type'] == pond_type])}")
        #save to csv
        pond_outliers.to_csv(f'fifty_one/measurements/results/analysis/outliers_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.csv', index=False)

  


    #



    # Get min and max for diagonal lines
    min_val = min(df['mean_length'].min(), df['Length_fov(mm)'].min())
    max_val = max(df['mean_length'].max(), df['Length_fov(mm)'].max())
    x = np.linspace(min_val, max_val, 100)

    #std

    # Create scatter plots for each pond type
    for pond_type in df['Pond_Type'].unique():
        print(f"Mean length for {pond_type}: {df[df['Pond_Type'] == pond_type]['mean_length'].mean()}")
        print(f"Std length for {pond_type}: {df[df['Pond_Type'] == pond_type]['std_length'].mean()}")
        print(f"Number of prawns in {pond_type}: {len(df[df['Pond_Type'] == pond_type])}")
        df_pond = df[df['Pond_Type'] == pond_type]

        min_val = min(df_pond['mean_length'].min(), df_pond['Length_fov(mm)'].min())
        max_val = max(df_pond['mean_length'].max(), df_pond['Length_fov(mm)'].max())
        x = np.linspace(min_val, max_val, 100)

        median_mad = df_pond['std_length'].mean()

        # Create scatter plot
        scatter = go.Scatter(
            x=df_pond['mean_length'],
            y=df_pond['Length_fov(mm)'],
            mode='markers',
            name='Measurements',
            text=df_pond.apply(lambda row: f"Image: {row['Label']}<br>Prawn ID: {row['PrawnID']}", axis=1),
            hoverinfo='text+x+y',
            marker=dict(
                color='#ff7f0e',
                opacity=0.7
            )
        )

       

        # Create diagonal lines
        diag = go.Scatter(x=x, y=x, mode='lines', name='Perfect match (y=x)', 
                        line=dict(color='black', dash='dash'))
        upper = go.Scatter(x=x, y=x + median_mad, mode='lines', 
                        name=f'mean + Std (uncertainty) ({median_mad:.2f} mm)',
                        line=dict(color='#2c7fb8', dash='dash'))
        lower = go.Scatter(x=x, y=x - median_mad, mode='lines',
                        name=f'mean - Std (uncertainty) ({median_mad:.2f} mm)', 
                        line=dict(color='#2c7fb8', dash='dash'))

        # Calculate percentage within bounds for this pond type
        within_bounds = ((df_pond['Length_fov(mm)'] <= df_pond['mean_length'] + median_mad) & 
                        (df_pond['Length_fov(mm)'] >= df_pond['mean_length'] - median_mad)).mean() * 100

        # Create layout
        layout = go.Layout(
            title=f'Model Measurements vs mean Values with Std (uncertainty) Boundaries - {pond_type}',
            xaxis_title='mean of manual measurements (mm)',
            yaxis_title='Model Measurements  (mm)',
            showlegend=True,
            width=800,
            height=800
        )

        # Create figure and add traces
        fig = go.Figure(data=[scatter, diag, upper, lower], layout=layout)

        # Update layout for equal aspect ratio
        fig.update_layout(
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1
            )
        )

        fig.write_html(f"fifty_one/measurements/results/analysis/model_vs_mean_measurements_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html")

        # Create combined model vs mean plot with all pond types
        if pond_type == df['Pond_Type'].unique()[-1]:  # Only create combined plot after processing all pond types
            # Create figure with subplots
            combined_fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[f'Model vs Mean Measurements - {pt}' for pt in df['Pond_Type'].unique()],
                horizontal_spacing=0.1
            )
            
            # Process each pond type
            for idx, pond_type in enumerate(df['Pond_Type'].unique(), 1):
                df_pond = df[df['Pond_Type'] == pond_type]
                
                # Calculate min and max for diagonal lines
                min_val = min(df_pond['mean_length'].min(), df_pond['Length_fov(mm)'].min())
                max_val = max(df_pond['mean_length'].max(), df_pond['Length_fov(mm)'].max())
                x = np.linspace(min_val, max_val, 100)
                
                median_mad = df_pond['std_length'].mean()
                
                # Add scatter plot
                combined_fig.add_trace(
                    go.Scatter(
                        x=df_pond['mean_length'],
                        y=df_pond['Length_fov(mm)'],
                        mode='markers',
                        name='Measurements',
                        text=df_pond.apply(lambda row: f"Image: {row['Label']}<br>Prawn ID: {row['PrawnID']}", axis=1),
                        hoverinfo='text+x+y',
                        marker=dict(
                            color='#ff7f0e',
                            opacity=0.7
                        ),
                        showlegend=False
                    ),
                    row=1, col=idx
                )
                
                # Add diagonal lines
                combined_fig.add_trace(
                    go.Scatter(
                        x=x, y=x,
                        mode='lines',
                        name='Perfect match (y=x)',
                        line=dict(color='black', dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=idx
                )
                
                combined_fig.add_trace(
                    go.Scatter(
                        x=x, y=x + median_mad,
                        mode='lines',
                        name=f'mean + Std ({median_mad:.2f} mm)',
                        line=dict(color='#2c7fb8', dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=idx
                )
                
                combined_fig.add_trace(
                    go.Scatter(
                        x=x, y=x - median_mad,
                        mode='lines',
                        name=f'mean - Std ({median_mad:.2f} mm)',
                        line=dict(color='#2c7fb8', dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=idx
                )
                
                # Add custom legend using annotations
                combined_fig.add_annotation(
                    x=min_val, y=max_val,
                    xref=f'x{idx}', yref=f'y{idx}',
                    text=f'<b>Legend:</b><br>• Measurements<br>• Perfect match (y=x)<br>• mean + Std ({median_mad:.2f} mm)<br>• mean - Std ({median_mad:.2f} mm)',
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=4,
                    align='left',
                    xanchor='left',
                    yanchor='top'
                )
                
                # Update axes for equal aspect ratio
                combined_fig.update_xaxes(
                    title_text='Mean of Manual Measurements (mm)',
                    scaleanchor=f'y{idx}',
                    scaleratio=1,
                    row=1, col=idx
                )
                combined_fig.update_yaxes(
                    title_text='Model Measurements (mm)',
                    row=1, col=idx
                )
            
            # Update layout for combined plot
            combined_fig.update_layout(
                title_text='Model vs Mean Measurements Across All Pond Types',
                height=600,
                width=1800,
                showlegend=False  # Disable the main legend
            )
            
            # Save combined plot
            combined_fig.write_html(f"fifty_one/measurements/results/analysis/model_vs_mean_measurements_combined_{args.type}_{args.weights_type}_{args.error_size}.html")

        #add a trace to the figure to show the points that are within the std bounds


           # Calculate error components
        df_pond['pixel_error_contribution'] = abs(df_pond['diff_pixels'] / df_pond['mean_pixels'] * 100)
        df_pond['scale_error_contribution'] = abs(df_pond['diff_scale'] / df_pond['mean_scale'] * 100)
        
        # Calculate actual impact in millimeters with signs preserved
        df_pond['pixel_error_mm'] = (df_pond['mean_pixels'] - df_pond['pred_Distance_pixels']) * (1/df_pond['mean_scale']) * 10
        df_pond['scale_error_mm'] = df_pond['mean_pixels'] * (1/df_pond['mean_scale'] - 1/df_pond['pred_scale']) * 10
        
        # Calculate total error
        df_pond['total_error_mm'] = df_pond['pixel_error_mm'] + df_pond['scale_error_mm']



        df_pond['within_std'] = (
            (df_pond['Length_fov(mm)'] <= df_pond['mean_length'] + df_pond['std_length'].mean()) &
            (df_pond['Length_fov(mm)'] >= df_pond['mean_length'] - df_pond['std_length'].mean())
        )
        df_pond['outside_std'] = ~df_pond['within_std']


        df_pond['MAE'] = abs(df_pond['Length_fov(mm)'] - df_pond['mean_length'])
        from statsmodels import robust
        mae_mad = robust.mad(df_pond['MAE'])
        print(f"MAE for {pond_type}: , median: {df_pond['MAE'].median()}, mad: {mae_mad}, min: {df_pond['MAE'].min()}, max: {df_pond['MAE'].max()}")




        df_pond['MARE'] = abs(df_pond['Length_fov(mm)'] - df_pond['mean_length'])/df_pond['mean_length'] * 100
        mare_mad = robust.mad(df_pond['MARE'])      #median absolute deviation
        print(f"MARE for {pond_type}: , median: {df_pond['MARE'].median()}, mad: {mare_mad}, min: {df_pond['MARE'].min()}, max: {df_pond['MARE'].max()}")






        
        # For points outside std, determine which error source "sets the tone" (i.e., dominates the total error direction)
        # If pixel and scale error have the same sign, the dominant one is the one with the larger absolute value
        # If pixel and scale error have opposite signs (i.e., partially cancel), the one whose sign matches the total error sets the tone
        # If both are very small, label as 'Both'
        def determine_error_type(row):
            if row['MARE'] <= 5:
                return 'Within 5% MARE'
            pixel_err = row['pixel_error_mm']
            scale_err = row['scale_error_mm']
            total_err = row['total_error_mm']
            # If both errors are very small, label as 'Both'
            if abs(pixel_err) < 1e-6 and abs(scale_err) < 1e-6:
                return 'Both'
            # If both errors have the same sign, the larger one sets the tone
            if (pixel_err >= 0 and scale_err >= 0) or (pixel_err <= 0 and scale_err <= 0):
                if abs(pixel_err) > abs(scale_err):
                    return 'Pixel Error'
                elif abs(scale_err) > abs(pixel_err):
                    return 'Scale Error'
                else:
                    return 'Both'
            # If errors have opposite signs, the one whose sign matches the total error sets the tone
            if abs(total_err) < 1e-6:
                return 'Both'
            if (total_err > 0 and pixel_err > 0) or (total_err < 0 and pixel_err < 0):
                return 'Pixel Error'
            elif (total_err > 0 and scale_err > 0) or (total_err < 0 and scale_err < 0):
                return 'Scale Error'
            else:
                return 'Both'

        df_pond['error_type'] = df_pond.apply(determine_error_type, axis=1)

        # Percentage of points with MARE > 5% by error type
        beyond_5_df = df_pond[df_pond['MARE'] > 5]
        beyond_5_counts = beyond_5_df['error_type'].value_counts()
        print(f"\nFor {pond_type}:")
        print("Percentage of points with MARE > 5% by dominant error type:")
        if len(beyond_5_df) > 0:
            print((beyond_5_counts / len(beyond_5_df)) * 100)
        else:
            print("No points with MARE > 5%.")


        #if pixel error check if the annotation pixel error is smaller within those with pixel error
        pixel_error_df = df_pond[df_pond['error_type'].isin(['Pixel Error', 'Both'])]


        #ground-truth pixel mm error


        #check if the annotation pixel error is smaller
        pixel_error_df['annotation_pixel_error'] = abs(pixel_error_df['pred_Distance_pixels'] - pixel_error_df['Length_ground_truth_annotation_pixels'])
        #check if the annotation pixel error is smaller

        #annotation pixel error in mm
        pixel_error_df['annotation_pixel_error_mm'] = pixel_error_df['annotation_pixel_error'] * (1/pixel_error_df['mean_scale']) * 10

        pixel_error_df['annotation_pixel_error_smaller'] = pixel_error_df['annotation_pixel_error'] < pixel_error_df['pixel_error_mm']
        #percentage of points where the annotation pixel error is smaller
        print(f"Percentage of points where the annotation pixel error is smaller: {pixel_error_df['annotation_pixel_error_smaller'].mean()*100:.2f}%")
        

        #out of the scale error how much come from the same image name
        scale_error_df = df_pond[df_pond['error_type'].isin(['Scale Error', 'Both'])]

        # Calculate the percentage of scale error points that share the same image (Label) with at least one other scale error point
        if not scale_error_df.empty and 'Label' in scale_error_df:
            label_counts = scale_error_df['Label'].value_counts()
            print(label_counts)
            
            # Only consider labels that appear more than once among scale error points
            shared_label_count = (scale_error_df['Label'].isin(label_counts[label_counts > 1].index)).sum()
            percent_shared = 100 * shared_label_count / len(scale_error_df)
            print(f"Percentage of scale error points that share the same image (Label) with at least one other scale error point: {percent_shared:.2f}%")
        

        # Optionally, print summary counts for analysis
        # Updated: Print dominant error type counts for points with MARE > 5%
        mare_gt5_counts = df_pond[df_pond['MARE'] > 5]['error_type'].value_counts()
        print(f"\nFor {pond_type}:")
        print("Points with MARE > 5% by dominant error type:")
        print(mare_gt5_counts)




        fig.add_trace(go.Scatter(
            x=df_pond['mean_length'],
            y=df_pond['Length_fov(mm)'],
            mode='markers',
            marker=dict(
                color=np.where(
                    df_pond['within_std'],
                    'green',
                    df_pond['error_type'].map({'Pixel Error': 'red', 'Scale Error': 'orange'}).fillna('gray')
                )
            ),
            name='Model vs Manual',
            customdata=np.stack([df_pond['Label'], df_pond['PrawnID']], axis=-1) if 'Label' in df_pond and 'PrawnID' in df_pond else None,
            hovertemplate="Image: %{customdata[0]}<br>Prawn ID: %{customdata[1]}<br>Mean Length: %{x:.2f} mm<br>Model Length: %{y:.2f} mm<extra></extra>"
        ))

        fig.write_html(f"fifty_one/measurements/results/analysis/with_std_bounds_colors_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html")



        



        # Plot manual measurement means with error bars and model predictions for each prawn
        # Convert to Plotly with hover of mean±std, image name, prawnid

        # Sort by mean_length for better visualization (optional)
        sorted_df = df_pond.sort_values('mean_length').reset_index(drop=True)

        # Adaptive spacing: use a nonlinear function to increase spacing for larger datasets
        n_points = len(sorted_df)
        # Use uniform spacing for x-axis
        if n_points > 1:
            max_x = 800
            x = np.linspace(0, max_x, n_points)
        else:
            x = sorted_df.index.to_numpy()

        y_manual = sorted_df['mean_length']
        yerr_manual = sorted_df['std_length']
        y_model = sorted_df['Length_fov(mm)']
        labels = sorted_df['Label'] if 'Label' in sorted_df else [''] * len(sorted_df)
        prawnids = sorted_df['PrawnID'] if 'PrawnID' in sorted_df else [''] * len(sorted_df)

        # Manual measurements with error bars
        manual_trace = go.Scatter(
            x=x,
            y=y_manual,
            error_y=dict(
                type='data',
                array=yerr_manual,
                visible=True,
                color='rgba(31,119,180,0.5)',
                thickness=2,
                width=8
            ),
            mode='markers',
            marker=dict(
                color='rgba(31,119,180,1)',
                size=10,
                symbol='circle'
            ),
            name='Manual Mean ± Std',
            customdata=list(
                zip(
                    y_manual,
                    yerr_manual,
                    labels,
                    prawnids
                )
            ),
            hovertemplate=(
                "Index: %{x}<br>"
                "Manual Mean: %{customdata[0]:.2f} mm<br>"
                "Std: %{customdata[1]:.2f} mm<br>"
                "Image: %{customdata[2]}<br>"
                "PrawnID: %{customdata[3]}<extra></extra>"
            )
        )

        # Model predictions
        model_trace = go.Scatter(
            x=x,
            y=y_model,
            mode='markers',
            marker=dict(
                color='orange',
                size=10,
                symbol='x'
            ),
            name='Model Prediction',
            customdata=list(
                zip(
                    y_model,
                    labels,
                    prawnids
                )
            ),
            hovertemplate=(
                "Index: %{x}<br>"
                "Model Prediction: %{customdata[0]:.2f} mm<br>"
                "Image: %{customdata[1]}<br>"
                "PrawnID: %{customdata[2]}<extra></extra>"
            )
        )

        fig = go.Figure([manual_trace, model_trace])
        fig.update_layout(
            title=f'Manual Measurement Means (±Std) and Model Predictions - {pond_type}',
            xaxis_title='Prawn Index (spaced)',
            yaxis_title='Length (mm)',
            legend=dict(x=0.01, y=0.99),
            width=1000,
            height=500,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        fig.write_html(f"fifty_one/measurements/results/analysis/error_bars_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html")

        # Calculate error components
        df_pond['pixel_error_contribution'] = abs(df_pond['diff_pixels'] / df_pond['mean_pixels'] * 100)
        df_pond['scale_error_contribution'] = abs(df_pond['diff_scale'] / df_pond['mean_scale'] * 100)
        
        # Calculate actual impact in millimeters with signs preserved
        df_pond['pixel_error_mm'] = (df_pond['mean_pixels'] - df_pond['pred_Distance_pixels']) * (1/df_pond['mean_scale']) * 10
        df_pond['scale_error_mm'] = df_pond['mean_pixels'] * (1/df_pond['mean_scale'] - 1/df_pond['pred_scale']) * 10
        
        # Calculate total error
        df_pond['total_error_mm'] = df_pond['pixel_error_mm'] + df_pond['scale_error_mm']

        # Create stacked bar chart for points outside standard deviation bounds

        within_std_bounds = (
            (df_pond['Length_fov(mm)'] <= df_pond['mean_length'] + df_pond['std_length']) &
            (df_pond['Length_fov(mm)'] >= df_pond['mean_length'] - df_pond['std_length'])
        )
        outside_std_bounds = ~within_std_bounds

        # Sort by image name (Label)
        error_points = df_pond[outside_std_bounds].reset_index(drop=True)
        error_points = error_points.sort_values('Label', ascending=True).reset_index(drop=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Pixel Error',
            x=error_points.index,
            y=error_points['pixel_error_mm'],  # Already has correct sign
            marker_color='#1f77b4',
            hovertemplate="Image: %{customdata[0]}<br>" +
                         "Prawn ID: %{customdata[1]}<br>" +
                         "Mean Length: %{customdata[2]:.1f}mm<br>" +
                         "model Length: %{customdata[3]:.1f}mm<br>" +
                         "Pixel Error: %{y:.1f}mm<br>" +
                         "Impact: %{customdata[4]:.1f}%<extra></extra>",
            customdata=error_points[['Label', 'PrawnID', 'mean_length', 'Length_fov(mm)']].values
        ))
        fig.add_trace(go.Bar(
            name='Scale Error', 
            x=error_points.index,
            y=error_points['scale_error_mm'],  # Already has correct sign
            marker_color='#ff7f0e',
            hovertemplate="Image: %{customdata[0]}<br>" +
                         "Prawn ID: %{customdata[1]}<br>" +
                         "Mean Length: %{customdata[2]:.1f}mm<br>" +
                         "model Length: %{customdata[3]:.1f}mm<br>" +
                         "Scale Error: %{y:.1f}mm<br>" +
                         "Impact: %{customdata[4]:.1f}%<extra></extra>",
            customdata=error_points[['Label', 'PrawnID', 'mean_length', 'Length_fov(mm)']].values
        ))

        # Update layout
        fig.update_layout(
            barmode='relative',  # Shows bars side by side
            title=f'Error Components Analysis - {pond_type}',
            xaxis_title='Measurement Index',
            yaxis_title='Error (mm)',
            showlegend=True,
            width=1000,
            height=600
        )

        fig.write_html(f"fifty_one/measurements/results/analysis/error_components_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html")

        #create combined graph with the 3 pond types
        if pond_type == df['Pond_Type'].unique()[-1]:  # Only create combined plot after processing all pond types
            # Create figure with subplots
            combined_fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[f'Error Components Analysis - {pt}' for pt in df['Pond_Type'].unique()],
                horizontal_spacing=0.1
            )
            
            # Process each pond type
            for idx, pond_type in enumerate(df['Pond_Type'].unique(), 1):
                df_pond = df[df['Pond_Type'] == pond_type]
                
                # Calculate error components
                df_pond['pixel_error_mm'] = (df_pond['mean_pixels'] - df_pond['pred_Distance_pixels']) * (1/df_pond['mean_scale']) * 10
                df_pond['scale_error_mm'] = df_pond['mean_pixels'] * (1/df_pond['mean_scale'] - 1/df_pond['pred_scale']) * 10
                
                # Calculate total error
                df_pond['total_error_mm'] = df_pond['pixel_error_mm'] + df_pond['scale_error_mm']
                
                # Get points outside standard deviation bounds
                within_std_bounds = (
                    (df_pond['Length_fov(mm)'] <= df_pond['mean_length'] + df_pond['std_length']) &
                    (df_pond['Length_fov(mm)'] >= df_pond['mean_length'] - df_pond['std_length'])
                )
                outside_std_bounds = ~within_std_bounds
                
                # Sort by image name (Label)
                error_points = df_pond[outside_std_bounds].reset_index(drop=True)
                error_points = error_points.sort_values('Label', ascending=True).reset_index(drop=True)
                
                # Add traces for this pond type
                combined_fig.add_trace(
                    go.Bar(
                        name='Pixel Error',
                        x=error_points.index,
                        y=error_points['pixel_error_mm'],
                        marker_color='#1f77b4',
                        hovertemplate="Image: %{customdata[0]}<br>" +
                                    "Prawn ID: %{customdata[1]}<br>" +
                                    "Mean Length: %{customdata[2]:.1f}mm<br>" +
                                    "Model Length: %{customdata[3]:.1f}mm<br>" +
                                    "Pixel Error: %{y:.1f}mm<br>" +
                                    "Impact: %{customdata[4]:.1f}%<extra></extra>",
                        customdata=error_points[['Label', 'PrawnID', 'mean_length', 'Length_fov(mm)']].values
                    ),
                    row=1, col=idx
                )
                
                combined_fig.add_trace(
                    go.Bar(
                        name='Scale Error',
                        x=error_points.index,
                        y=error_points['scale_error_mm'],
                        marker_color='#ff7f0e',
                        hovertemplate="Image: %{customdata[0]}<br>" +
                                    "Prawn ID: %{customdata[1]}<br>" +
                                    "Mean Length: %{customdata[2]:.1f}mm<br>" +
                                    "Model Length: %{customdata[3]:.1f}mm<br>" +
                                    "Scale Error: %{y:.1f}mm<br>" +
                                    "Impact: %{customdata[4]:.1f}%<extra></extra>",
                        customdata=error_points[['Label', 'PrawnID', 'mean_length', 'Length_fov(mm)']].values
                    ),
                    row=1, col=idx
                )
            
            # Update layout for combined plot
            combined_fig.update_layout(
                barmode='relative',  # Shows bars side by side
                title_text='Error Components Analysis Across All Pond Types',
                height=600,  # Reduced height since plots are side by side
                width=1800,  # Increased width to accommodate side-by-side plots
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                )
            )
            
            # Update y-axis labels for all subplots
            for i in range(1, 4):
                combined_fig.update_yaxes(title_text="Error (mm)", row=1, col=i)
                combined_fig.update_xaxes(title_text="Measurement Index", row=1, col=i)
            
            # Save combined plot
            combined_fig.write_html(f"fifty_one/measurements/results/analysis/error_components_combined_{args.type}_{args.weights_type}_{args.error_size}.html")











        #create the same for mean aground truth pixels and Length_ground_truth_annotation_pixels
        df_pond['pixel_error_mm_ground_truth'] = abs(df_pond['Length_ground_truth_annotation_pixels'] - df_pond['pred_Distance_pixels']) 

        
        # Calculate actual impact in millimeters with signs preserved
        df_pond['pixel_error_mm_ground_truth'] = (df_pond['Length_ground_truth_annotation_pixels'] - df_pond['pred_Distance_pixels']) * (1/df_pond['mean_scale']) * 10
        df_pond['scale_error_mm_ground_truth'] = df_pond['Length_ground_truth_annotation_pixels'] * (1/df_pond['mean_scale'] - 1/df_pond['pred_scale']) * 10
        
        # Calculate total error
        df_pond['total_error_mm_ground_truth'] = df_pond['pixel_error_mm_ground_truth'] + df_pond['scale_error_mm_ground_truth']

        # Create stacked bar chart for points outside standard deviation bounds
        error_points = df_pond[outside_std_bounds].reset_index(drop=True)
        error_points = error_points.sort_values('total_error_mm_ground_truth', key=abs, ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Pixel Error',
            x=error_points.index,
            y=error_points['pixel_error_mm_ground_truth'],  # Already has correct sign
            marker_color='#1f77b4',
            hovertemplate="Image: %{customdata[0]}<br>" +
                         "Prawn ID: %{customdata[1]}<br>" +
                         "Mean Length: %{customdata[2]:.1f}mm<br>" +
                         "model Length: %{customdata[3]:.1f}mm<br>" +
                         "Pixel Error: %{y:.1f}mm<br>" +
                         "Impact: %{customdata[4]:.1f}%<extra></extra>",
            customdata=error_points[['Label', 'PrawnID', 'mean_annotation_length', 'Length_fov(mm)']].values
        ))
        # fig.add_trace(go.Bar(
        #     name='Scale Error', 
        #     x=error_points.index,
        #     y=error_points['scale_error_mm_ground_truth'],  # Already has correct sign
        #     marker_color='#ff7f0e',
        #     hovertemplate="Image: %{customdata[0]}<br>" +
        #                  "Prawn ID: %{customdata[1]}<br>" +
        # #                  "Mean Length: %{customdata[2]:.1f}mm<br>" +
        # #                  "model Length: %{customdata[3]:.1f}mm<br>" +
        # #                  "Scale Error: %{y:.1f}mm<br>" +
        # #                  "Impact: %{customdata[4]:.1f}%<extra></extra>",
        # #     customdata=error_points[['Label', 'PrawnID', 'mean_length', 'Length_fov(mm)']].values
        # # ))

        # Update layout
        fig.update_layout(
            barmode='relative',  # Shows bars side by side
            title=f'Error Components Analysis - {pond_type}',
            xaxis_title='Measurement Index',
            yaxis_title='Error (mm)',
            showlegend=True,
            width=1000,
            height=600
        )

        fig.write_html(f"fifty_one/measurements/results/analysis/error_components_ground_truth_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html")








        # # Calculate error components
        # df_pond['pixel_error_contribution'] = abs(df_pond['diff_pixels'] / df_pond['mean_pixels'] * 100)
        # df_pond['scale_error_contribution'] = abs(df_pond['diff_scale'] / df_pond['mean_scale'] * 100)

        # # Recalculate within_std_bounds for this specific pond type
        # pond_within_std_bounds = ((df_pond['Length_fov(mm)'] <= df_pond['mean_length'] + median_mad) & 
        #                         (df_pond['Length_fov(mm)'] >= df_pond['mean_length'] - median_mad))


        # #mean length
        

        # # Create stacked bar chart for points with errors
        # error_points = df_pond[~pond_within_std_bounds].reset_index(drop=True)
        # error_points['mean_length'] = df_pond['mean_pixels']*(1/(df_pond['mean_scale']))*10
        # # Calculate total error percentage as MPE
        # error_points['total_error_pct'] =  df_pond['pixel_error_contribution']+df_pond['scale_error_contribution']

        # # Sort by total error
        # error_points = error_points.sort_values('total_error_pct', ascending=False)

        # # Calculate mean MPE
        # error_points['mean_mpe'] = abs(error_points['Length_fov(mm)'] - error_points['mean_length'])/error_points['mean_length']*100
        # fig = go.Figure()
        # fig.add_trace(go.Bar(
        #     name='Pixel Error Contribution',
        #     x=error_points.index,
        #     y=error_points['pixel_error_contribution'],
        #     marker_color='#1f77b4',
        #     hovertemplate="Image: %{customdata[0]}<br>" +
        #                  "Prawn ID: %{customdata[1]}<br>" +
        #                  "Pixel Error: %{y:.1f}%<br>" +
        #                  "Mean MPE: %{customdata[2]:.1f}%<extra></extra>",
        #     customdata=error_points[['Label', 'PrawnID', 'mean_mpe']].values
        # ))
        # fig.add_trace(go.Bar(
        #     name='Scale Error Contribution', 
        #     x=error_points.index,
        #     y=error_points['scale_error_contribution'],
        #     marker_color='#ff7f0e',
        #     hovertemplate="Image: %{customdata[0]}<br>" +
        #                  "Prawn ID: %{customdata[1]}<br>" +
        #                  "Scale Error: %{y:.1f}%<br>" +
        #                  "Mean MPE: %{customdata[2]:.1f}%<extra></extra>",
        #     customdata=error_points[['Label', 'PrawnID', 'mean_mpe']].values
        # ))

        # # Update layout
        # fig.update_layout(
        #     barmode='stack',
        #     title=f'Error Components Analysis - {pond_type}',
        #     xaxis_title='Measurement Index',
        #     yaxis_title='Error Contribution (%)',
        #     showlegend=True,
        #     width=1000,
        #     height=600
        # )

    


        # fig.write_html(f"fifty_one/measurements/results/analysis/stacked_error_components_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html")

       
    # Compare Length_fov with mean annotation length for each pond type
    for pond_type in df['Pond_Type'].unique():
        df_pond = df[df['Pond_Type'] == pond_type]

        df_pond['mpe']=abs(df_pond['Length_fov(mm)'] - df_pond['mean_annotation_length'])/df_pond['mean_annotation_length'] * 100

        #without outliers in mean percentage error
        df_pond = df_pond[df_pond['mpe'] < df_pond['mpe'].std() * 3]
        
        print(f"\nLength Comparison Stats for {pond_type}:")
        print(f"Mean Length_fov: {df_pond['Length_fov(mm)'].mean():.2f} mm")
        print(f"Mean annotation length: {df_pond['mean_annotation_length'].mean():.2f} mm")
        
        # Calculate percentage difference
        pct_diff = abs(df_pond['Length_fov(mm)'] - df_pond['mean_annotation_length'])/df_pond['mean_annotation_length'] * 100
        print(f"Mean percentage difference: {pct_diff.mean():.2f}%")
        print(f"Median percentage difference: {pct_diff.median():.2f}%")
        print(f"Std dev of percentage difference: {pct_diff.std():.2f}%")
        
        # Calculate correlation
        correlation = df_pond['Length_fov(mm)'].corr(df_pond['mean_annotation_length'])
        print(f"Correlation coefficient: {correlation:.3f}")








    # Analyze points for each pond type
    for pond_type in df['Pond_Type'].unique():
        df_pond = df[df['Pond_Type'] == pond_type]
        
        # Get counts and percentages for this pond type
        slope, intercept, r_value, p_value, std_err = stats.linregress(
        df_pond['mean_length'], df_pond['Length_fov(mm)']
        )


        avg_std = df_pond['std_length'].mean()

        df_pond['predicted_length'] = slope * df_pond['mean_length'] + intercept
        within_std_bounds = (
            (df_pond['Length_fov(mm)'] <= df_pond['mean_length'] + df_pond['std_length'].mean()) &
            (df_pond['Length_fov(mm)'] >= df_pond['mean_length'] - df_pond['std_length'].mean())
        )
        outside_std_bounds = ~within_std_bounds






        total_count = len(df_pond)
        within_count = len(df_pond[within_std_bounds])
        outside_count = len(df_pond[outside_std_bounds])
        
        print(f"\nPoints Analysis for {pond_type} ponds:")
        print(f"Points within std bounds: {within_count} out of {total_count} ({within_count/total_count*100:.1f}%)")
        print(f"Points outside std bounds: {outside_count} out of {total_count} ({outside_count/total_count*100:.1f}%)")
        
        # For points outside bounds, calculate distance beyond std boundary
        if outside_count > 0:
            df_outside = df_pond[outside_std_bounds]
            distance_beyond_std = df_outside.apply(lambda row: 
                abs(row['Length_fov(mm)'] - (row['mean_length'] + std)) 
                if row['Length_fov(mm)'] > row['mean_length'] + std
                else abs(row['Length_fov(mm)'] - (row['mean_length'] - std)), axis=1)
                
            mean_distance_beyond = distance_beyond_std.mean()
            median_distance_beyond = distance_beyond_std.median()
            std_distance_beyond = distance_beyond_std.std()
            #mad distance beyond std
            mad_distance_beyond = stats.median_abs_deviation(distance_beyond_std)
            
            print("\nDistance Beyond Std Bounds:")
            print(f"Mean distance beyond std: {mean_distance_beyond:.2f} mm")
            print(f"Median distance beyond std: {median_distance_beyond:.2f} mm")
            print(f"Standard deviation of distance beyond std: {std_distance_beyond:.2f} mm")
            print(f"MAD of distance beyond std: {mad_distance_beyond:.2f} mm")
            # Calculate percentage beyond std bounds
            percent_beyond_std = (distance_beyond_std / abs(df_outside.apply(lambda row: 
                row['mean_length'] + std if row['Length_fov(mm)'] > row['mean_length'] 
                else row['mean_length'] - std, axis=1)))*100
            mean_percent_beyond = percent_beyond_std.median()
            std_percent_beyond = percent_beyond_std.std()
            mad_percent_beyond = stats.median_abs_deviation(percent_beyond_std)
            
            print(f"Mean percentage beyond std: {mean_percent_beyond:.2f}%")
            print(f"Standard deviation of percentage beyond std: {std_percent_beyond:.2f}%")
            print(f"MAD of percentage beyond std: {mad_percent_beyond:.2f}%")
        else:
            print("\nNo points outside std bounds")








    df['mpe_mean_length'] = abs(df['mean_length'] - df['Length_fov(mm)']) / df['mean_length'] * 100


    # Calculate mean and standard deviation for expert measurements
    df['Mean_Length'] = df[['Length_1', 'Length_2', 'Length_3']].mean(axis=1)
    df['Std_Length'] = df[['Length_1', 'Length_2', 'Length_3']].std(axis=1)

    # Take the closest measurement to the predicted value
    df['Closest_To_Prediction'] = df[['Length_1', 'Length_2', 'Length_3']].apply(
        lambda x: x.iloc[(x - df['Length_fov(mm)']).abs().argmin()], axis=1)
    
    # Absolute difference between closest measurement and predicted value
    df['Diff_Closest_Pred'] = (df['Closest_To_Prediction'] - df['Length_fov(mm)']).abs()

    # # Justification based on whether it's within 1 standard deviation
    # df['Justified'] = df['Diff_Closest_Pred'] <= df['Std_Length']
    # print('========Justified========')
    # print(df['Justified'].value_counts())


    df['MPE_annotation_length_1'] = abs(df['annotation_length_1'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['MPE_annotation_length_2'] = abs(df['annotation_length_2'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['MPE_annotation_length_3'] = abs(df['annotation_length_3'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100

    df['mae_length1'] = abs(df['Length_1']-df['Length_fov(mm)'])
    df['mae_length2'] = abs(df['Length_2']-df['Length_fov(mm)'])
    df['mae_length3'] = abs(df['Length_3']-df['Length_fov(mm)'])


    df['mae_with_sign_1'] =  df['Length_1']- df['Length_fov(mm)'] 
    df['mae_with_sign_2'] = df['Length_2']- df['Length_fov(mm)']
    df['mae_with_sign_3'] = df['Length_3']- df['Length_fov(mm)']   

    df['mpe_with_sign_1'] = (df['Length_1'] - df['Length_fov(mm)']) / df['Length_1'] * 100
    df['mpe_with_sign_2'] = (df['Length_2'] - df['Length_fov(mm)']) / df['Length_2'] * 100
    df['mpe_with_sign_3'] = (df['Length_3'] - df['Length_fov(mm)']) / df['Length_3'] * 100


    df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].min(axis=1)
    df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].min(axis=1)




    df['mae_annotation_length_1'] = abs(df['Length_fov(mm)'] - df['annotation_length_1'])
    df['mae_annotation_length_2'] = abs(df['Length_fov(mm)'] - df['annotation_length_2'])
    df['mae_annotation_length_3'] = abs(df['Length_fov(mm)'] - df['annotation_length_3'])
    
    df['mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_annotation_length_1', 'MPE_annotation_length_2', 'MPE_annotation_length_3']].min(axis=1)
    df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_annotation_length_1', 'mae_annotation_length_2', 'mae_annotation_length_3']].min(axis=1)
    
    df['avg_length'] = df[['Length_1', 'Length_2', 'Length_3']].mean(axis=1)
    
    if  args.error_size == 'min':    
    #     df['mae_annotation_length'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_annotation_length_1', 'mae_annotation_length_2', 'mae_annotation_length_3']].min(axis=1)
    #     df['mpe_annotation_length'] = df[['mpe_length1', 'mpe_length2', 'mpe_length3', 'mpe_annotation_length_1', 'mpe_annotation_length_2', 'mpe_annotation_length_3']].min(axis=1)
   
    # # Determine the minimum MPE across all three measurements for each row
    # This represents the best-case error for each measuremen
        #with annotation
        df[f'mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_annotation_length_1', 'MPE_annotation_length_2', 'MPE_annotation_length_3']].min(axis=1)
        df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_annotation_length_1', 'mae_annotation_length_2', 'mae_annotation_length_3']].min(axis=1)

        #without annotation
        df['mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].min(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].min(axis=1)


        #remove outliers statistically
        df = df[df['mpe'] < df['mpe'].std() * 3]
        df = df[df['mae'] < df['mae'].std() * 3]
        
        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].median(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].median(axis=1)
        
        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].min(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].min(axis=1)
        
        df['accounting_length'] = df.apply(
            lambda row: row['Length_1'] if row['mpe_with_sign_1'] == row['mpe'] else
                        (row['Length_2'] if row['mpe_with_sign_2'] == row['mpe'] else row['mpe_with_sign_3']),
            axis=1
)   
        
        df['mean_scale'] = df[['Scale_1', 'Scale_2', 'Scale_3']].mean(axis=1)
        

        #min annotation length 
        df['mean_annotation_length'] = df['Length_ground_truth_annotation_pixels'] / df['mean_scale'] * 10
    #mpe annotation length
        df['mpe_annotation'] = abs (df['mean_annotation_length'] - df['Length_fov(mm)'])/df['mean_annotation_length']*100

        df['mpe_annotation_length'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'mpe_annotation']].min(axis=1)




        df['mae_annotation'] = abs(df['mean_annotation_length'] - df['Length_fov(mm)'])


        df['mpe_annotation_length'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'mpe_annotation']].min(axis=1)

        df['mae_annotation_length'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_annotation']].min(axis=1)

 #show smoothed histogram of mae
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df['mae'], fill=True)
        plt.xlabel('Mean Absolute Error (mm)')
        plt.ylabel('Density')
        plt.title('Smoothed Distribution of Mean Absolute Error')
        # plt.show()

        #show smoothed histogram of mpe
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df['mpe'], fill=True)
        plt.xlabel('Mean Percentage Error (%)')
        plt.ylabel('Density')
        plt.title('Smoothed Distribution of Mean Percentage Error')
        # plt.show()




    elif args.error_size == 'median':

        df['median_annotation_length'] = df['Length_ground_truth_annotation_pixels'] / df['median_scale'] * 10

        df['MPE_median_annotation_length'] = abs(df['median_annotation_length'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
        df['mae_median_annotation_length'] = abs(df['median_annotation_length'] - df['Length_fov(mm)'])
        df['mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_median_annotation_length']].median(axis=1)
        df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_median_annotation_length']].median(axis=1)

        #without annotation
        
        
        
        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].median(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].median(axis=1)
        
        
        
        df['mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].median(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].median(axis=1)

        df['mpe'] =df[['mpe_with_sign']]
        df['mae'] =df[['mae_with_sign']]
        
    
        df['accounting_length'] = df.apply(
            lambda row: row['Length_1'] if row['mpe_with_sign_1'] == row['mpe_with_sign'] else
                        (row['Length_2'] if row['mpe_with_sign_2'] == row['mpe_with_sign'] else row['Length_3']),
            axis=1
)

        df['accounting_length_pixels'] = df.apply(
            lambda row: row['Length_1_pixels'] if row['mpe_with_sign_1'] == row['mpe_with_sign'] else   
                        (row['Length_2_pixels'] if row['mpe_with_sign_2'] == row['mpe_with_sign'] else row['Length_3_pixels']),
            axis=1
)
        

        df['accounting_scale'] = df.apply(
            lambda row: row['Scale_1'] if row['mpe_with_sign_1'] == row['mpe_with_sign'] else
                        (row['Scale_2'] if row['mpe_with_sign_2'] == row['mpe_with_sign'] else row['Scale_3']),
            axis=1
)

    elif args.error_size == 'mean':
        df['mean_scale'] = df[['Scale_1', 'Scale_2', 'Scale_3']].mean(axis=1)
        df['mean_annotation_length'] = df['Length_ground_truth_annotation_pixels'] / df['mean_scale'] * 10
        df['MPE_mean_annotation_length'] = abs(df['mean_annotation_length'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
        df['mae_mean_annotation_length'] = abs(df['mean_annotation_length'] - df['Length_fov(mm)'])
        df['mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_mean_annotation_length']].mean(axis=1)
        df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_mean_annotation_length']].mean(axis=1)
        
       
        #without annotation
        df['mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].mean(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].mean(axis=1)


       #df only outliers

        #remove outliers statistically 
        df = df[df['mpe'] < df['mpe'].std() * 3]
        df = df[df['mae'] < df['mae'].std() * 3]




         #show smoothed histogram of mae
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df['mae'], fill=True)
        plt.xlabel('Mean Absolute Error (mm)')
        plt.ylabel('Density')
        plt.title('Smoothed Distribution of Mean Absolute Error')
        # plt.show()

        #show smoothed histogram of mpe
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df['mpe'], fill=True)
        plt.xlabel('Mean Percentage Error (%)')
        plt.ylabel('Density')
        plt.title('Smoothed Distribution of Mean Percentage Error')
        # plt.show()

 #if row justitfiend take the min mpe as the mpe
        # df['mpe'] = df.apply(lambda row: min(row['MPE_length1'], row['MPE_length2'], row['MPE_length3']) if row['Justified'] else row['mpe'], axis=1)
        # df['mae'] = df.apply(lambda row: min(row['mae_length1'], row['mae_length2'], row['mae_length3']) if row['Justified'] else row['mae'], axis=1)

        # #flag justified
        # df['flag_justified'] = df['Justified']

        #in pond circlefemale less than 35

        # Filter out rows where Pond_Type is circle_female and mpe >= 35
        df = df[~((df['Pond_Type'] == 'circle_female') & (df['mpe'] >= 35))]
        
        # df['mpe']= df.apply(lambda row: row['mpe']<35 if row['Pond_Type'] == 'circle_female' else row['mpe'], axis=1)
        
        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].mean(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].mean(axis=1)
        
          
#annotation length based mean scale 
        df['mean_annotation_length'] = df['Length_ground_truth_annotation_pixels'] / df['mean_scale'] * 10
    #mpe annotation length
        df['mpe_annotation'] = abs(df['mean_annotation_length'] - df['Length_fov(mm)'])/df['Length_fov(mm)']*100

        df['mae_annotation'] = abs(df['mean_annotation_length'] - df['Length_fov(mm)'])


        df['mpe_annotation_length'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'mpe_annotation']].mean(axis=1)

        df['mae_annotation_length'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_annotation']].mean(axis=1)

        #if justified take that length as the best length
        

    elif args.error_size == 'max':

        
        #without annotation
        df['mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].max(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].max(axis=1)
        
        df['mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_annotation_length_1', 'MPE_annotation_length_2', 'MPE_annotation_length_3']].max(axis=1)
        df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_annotation_length_1', 'mae_annotation_length_2', 'mae_annotation_length_3']].max(axis=1)

        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].max(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].max(axis=1)

        df['accounting_length'] = df.apply(
            lambda row: row['Length_1'] if row['mpe_with_sign_1'] == row['mpe'] else
                        (row['Length_2'] if row['mpe_with_sign_2'] == row['mpe'] else row['mpe_with_sign_3']),
            axis=1
)
        
        df['mpe'] = df[['mpe_with_sign']]
        df['mae'] = df[['mae_with_sign']]




    #mean scale

    #make annotation_length




# ----- Best Length Determination -----

# Create a mask identifying which length measurement gave the minimum MPE
    min_mpe_mask = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].eq(df['mpe'], axis=0)

# Map column names to their corresponding indices for reference
    column_to_index = {
    'MPE_length1': 1,
    'MPE_length2': 2,
    'MPE_length3': 3
}

# Get the column name with minimum MPE for each row
    
    min_mpe_column = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].idxmin(axis=1)

# Map to the correct index (1, 2, or 3) for each measurement
    min_mpe_index = min_mpe_column.map(column_to_index)

# Use the index to get the corresponding best length measurement for each row
# This selects the length with minimum error for further analysis
    df['best_length'] = df.apply(lambda row: row[f'Length_{min_mpe_index[row.name]}'], axis=1)

# Similarly, get the best length in pixels
    df['best_length_pixels'] = df.apply(lambda row: row[f'Length_{min_mpe_index[row.name]}_pixels'], axis=1)

# ----- Pixel Error Calculations -----

# Calculate prediction scale (pixels per mm)
  
# Normalize expert measurements to pixels for comparison with predictions
# This converts the best expert measurement to the same pixel scale as predictions


    # df['flag_scale_error'] =(abs(df['pred_scale'] - df['Scale_1'])/df['Scale_1']*100 > 10) & (abs(df['pred_scale'] - df['Scale_2'])/df['Scale_2']*100 > 10 )& (abs(df['pred_scale'] - df['Scale_3'])/df['Scale_3']*100 > 10)
   
   #create table pred scale near scale_1, scale_2, scale_3 table


    
    #scale normalize
    df['scale_normalized_1'] = 1/(df['Scale_1']/10)
    df['scale_normalized_2'] = 1/(df['Scale_2']/10)
    df['scale_normalized_3'] = 1/(df['Scale_3']/10)


    # print(f'combined scale: {df["combined_scale"].describe()}')
    # print(f'scale_normalized_1: {df["scale_normalized_1"].describe()}')
    # print(f'scale_normalized_2: {df["scale_normalized_2"].describe()}')
    # print(f'scale_normalized_3: {df["scale_normalized_3"].describe()}')
   
        #avg scale error
    df['scale_error_1'] = abs (df['combined_scale'] - df['scale_normalized_1'])/df['scale_normalized_1']*100
    df['scale_error_2'] = abs(df['combined_scale'] - df['scale_normalized_2'])/df['scale_normalized_2']*100
    df['scale_error_3'] = abs(df['combined_scale'] - df['scale_normalized_3'])/df['scale_normalized_3']*100



     #scale diff
    df['scale_diff_1'] = abs(df['combined_scale'] - df['scale_normalized_1'])
    df['scale_diff_2'] = abs(df['combined_scale'] - df['scale_normalized_2'])
    df['scale_diff_3'] = abs(df['combined_scale'] - df['scale_normalized_3'])

    #avg scale diff
    df['avg_scale_diff'] = df[['scale_diff_1', 'scale_diff_2', 'scale_diff_3']].mean(axis=1)

    df['avg_scale_error'] = df[['scale_error_1', 'scale_error_2', 'scale_error_3']].mean(axis=1)
    if args.type=='carapace':
        df['flag_avg_scale_error'] = abs(df['avg_scale_error'])  > 15       
    else:
        df['flag_avg_scale_error'] = abs(df['avg_scale_error'])  > 15
   

    
   
    # df['mpe']=df['mpe_mean_length']

    
    # df['mae_mean_length']=abs(df['mean_length'] - df['Length_fov(mm)'])

    # df['mae']=df['mae_mean_length']
   
   
    # df['flag_scale_error'] = abs(df['pred_scale'] - df['accounting_scale'])/df['accounting_scale']*100 > 10
# Calculate minimum error in pixels (absolute difference between expert and prediction)
    # df['min_error_pixels'] = abs(df['expert_normalized_pixels'] - df['pred_Distance_pixels'])

    df['min_error_pixels']= df[['Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].min(axis=1)
# Calculate minimum MAPE in pixels (percentage error in pixel space)
    df['min_mape_pixels'] = df['min_error_pixels'] / df['pred_Distance_pixels'] * 100

# ----- Flag High Errors -----

# Create a flag for measurements with errors > 10%




# high error flag is above 5 mm




    if args.type=='carapace':
        df['high_error'] = abs(df['mae']) >3  
    else:
        df['high_error'] = abs(df['mae']) >10


# Calculate and display the percentage of measurements with high errors
    
    
    
    # high_error_pct = df['high_error'].mean() * 100
    # print(f"Percentage of measurements with errors > 10%: {high_error_pct:.1f}%")






# ----- Additional Pixel Differences -----

# Calculate absolute pixel differences between predicted and expert measurements
    # df['pred_pixels_diff'] = abs(df['pred_Distance_pixels'] - df['expert_normalized_pixels'])

# Calculate absolute pixel differences between ground truth annotation and prediction
    df['pred_pixel_gt_diff'] = abs(df['Length_ground_truth_annotation_pixels'] - df['pred_Distance_pixels'])

# Calculate absolute pixel differences for each expert measurement
    df['pixel_diff_1'] = abs(df['pred_Distance_pixels'] - df['Length_1_pixels']) 
    df['pixel_diff_2'] = abs(df['pred_Distance_pixels'] - df['Length_2_pixels'])
    df['pixel_diff_3'] = abs(df['pred_Distance_pixels'] - df['Length_3_pixels'])

# Calculate percentage differences for each measurement
    df['pixel_diff_pct_1'] = df['pixel_diff_1'] / df['Length_1_pixels'] * 100
    df['pixel_diff_pct_2'] = df['pixel_diff_2'] / df['Length_2_pixels'] * 100
    df['pixel_diff_pct_3'] = df['pixel_diff_3'] / df['Length_3_pixels'] * 100


    #avg pixel diff
    df['avg_pixel_diff'] = df[['pixel_diff_1', 'pixel_diff_2', 'pixel_diff_3']].mean(axis=1)
# ----- Flagging Pixel Errors -----

# Define threshold for high pixel percentage error
    if args.type=='carapace':
        pixel_pct_threshold = 10  # 10% threshold
    else:
        pixel_pct_threshold = 5 # 15% threshold

    # df['flag_accounting_pixel_error'] = abs(df['pred_Distance_pixels'] - df['accounting_length_pixels'])/df['Length_ground_truth_annotation_pixels']*100 > pixel_pct_threshold
    # Flag when all three measurements exceed the threshold
# This indicates consistent high error across all expert measurements
    df['flag_all_high_pixel_error'] = (
    (df['pixel_diff_pct_1'] > pixel_pct_threshold) & 
    (df['pixel_diff_pct_2'] > pixel_pct_threshold) & 
    (df['pixel_diff_pct_3'] > pixel_pct_threshold)
)

# Flag when any measurement exceeds the threshold
# This captures cases where at least one measurement has high error
    df['flag_any_high_pixel_error'] = (
    (df['pixel_diff_pct_1'] > pixel_pct_threshold) | 
    (df['pixel_diff_pct_2'] > pixel_pct_threshold) | 
    (df['pixel_diff_pct_3'] > pixel_pct_threshold)
)

#flag all smaller than 10%
    df['flag_all_small_pixel_error'] = (
    (df['pixel_diff_pct_1'] < pixel_pct_threshold) & 
    (df['pixel_diff_pct_2'] < pixel_pct_threshold) & 
    (df['pixel_diff_pct_3'] < pixel_pct_threshold)
)
    df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].median(axis=1)
    if args.error_size == 'median':
        df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].median(axis=1)
    elif args.error_size == 'mean':
        df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].mean(axis=1)
    elif args.error_size == 'min':
        df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].min(axis=1)
    elif args.error_size == 'max':
        df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].max(axis=1)

# Calculate average pixel percentage error across all three measurements
    df['avg_pixel_error_pct'] = (df['pixel_diff_pct_1'] + df['pixel_diff_pct_2'] + df['pixel_diff_pct_3']) / 3

# Flag high average pixel error
    df['flag_high_avg_pixel_error'] = abs(df['avg_pixel_error_pct'])  > pixel_pct_threshold

# ----- GT-Expert Pixel Differences -----

# Calculate absolute pixel differences between ground truth and each expert measurement
    df['gt_expert_diff_1'] =abs  (df['Length_ground_truth_annotation_pixels'] - df['Length_1_pixels'])
    df['gt_expert_diff_2'] =abs (df['Length_ground_truth_annotation_pixels'] - df['Length_2_pixels'])
    df['gt_expert_diff_3'] =abs (df['Length_ground_truth_annotation_pixels'] - df['Length_3_pixels'])

# Calculate percentage differences relative to expert measurements
    df['gt_expert_diff_pct_1'] = df['gt_expert_diff_1'] / df['Length_1_pixels'] * 100
    df['gt_expert_diff_pct_2'] = df['gt_expert_diff_2'] / df['Length_2_pixels'] * 100
    df['gt_expert_diff_pct_3'] = df['gt_expert_diff_3'] / df['Length_3_pixels'] * 100

    # df['flage_gt_expert_accounting_length_pixels'] = abs(df['Length_ground_truth_annotation_pixels'] - df['accounting_length_pixels'])/df['Length_ground_truth_annotation_pixels']*100 > 10

    #flag high differncen between min length and max length

    df['min_length'] = df[['Length_1', 'Length_2', 'Length_3']].min(axis=1)
    df['max_length'] = df[['Length_1', 'Length_2', 'Length_3']].max(axis=1)

    if args.type=='carapace':
        df['flag_high_diff_min_max_length'] = abs(df['min_length'] - df['max_length']) > 5
    else:
        df['flag_high_diff_min_max_length'] = abs(df['min_length'] - df['max_length']) > 5

# ----- Flagging GT-Expert Errors -----

# Define threshold for high GT-Expert pixel percentage error

    if args.type=='carapace':
        gt_expert_pct_threshold = 10 # 10% threshold
    else:
        gt_expert_pct_threshold = 5 # 15% threshold

# Flag when all three GT-Expert measurements exceed the threshold
    df['flag_all_high_gt_expert_error'] = (
    (df['gt_expert_diff_pct_1'] > gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_2'] > gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_3'] > gt_expert_pct_threshold)
)
    
    #all smaller than 10%
    df['flag_all_small_gt_expert_error'] = (
    (df['gt_expert_diff_pct_1'] < gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_2'] < gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_3'] < gt_expert_pct_threshold)
)

# Flag when any GT-Expert measurement exceeds the threshold
    df['flag_any_high_gt_expert_error'] = (
    (df['gt_expert_diff_pct_1'] > gt_expert_pct_threshold) | 
    (df['gt_expert_diff_pct_2'] > gt_expert_pct_threshold) | 
    (df['gt_expert_diff_pct_3'] > gt_expert_pct_threshold)
)

# Calculate average GT-Expert pixel percentage error
    df['avg_gt_expert_error_pct'] = (
    df['gt_expert_diff_pct_1'] + df['gt_expert_diff_pct_2'] + df['gt_expert_diff_pct_3']
) / 3

# Flag high average GT-Expert pixel error
    df['flag_high_avg_gt_expert_error'] = abs(df['avg_gt_expert_error_pct']) > gt_expert_pct_threshold

# ----- Potential Error Source Flags -----

# Define flags for potential error sources
# 1. High pixel difference between ground truth and expert (normalized)

# 2. High pixel difference between prediction and expert (percentage)

# 3. Low pose evaluation score (if available)
# Check which pose evaluation column exists in the dataset
    if 'pose_eval' in df.columns:
        df['flag_low_pose_eval'] = df['pose_eval'] < 0.95
    elif 'pose_eval_iou' in df.columns:
        df['flag_low_pose_eval'] = df['pose_eval_iou'] < 0.95
    else:
        df['flag_low_pose_eval'] = False
        print("Warning: No pose evaluation column found!")

# 4. High pixel difference between prediction and ground truth annotation
    df['flag_pred_gt_diff'] = abs(df['pred_pixel_gt_diff']/df['Length_ground_truth_annotation_pixels']*100) > 5

# ----- Flag Count and Multiple Error Images -----


# ----- Identify Images with All High Errors -----

# Get a complete list of unique image labels
    all_image_labels = df['Label'].unique()

# Count total measurements per image
    total_measurements_by_image = df.groupby('Label').size()

# Count high error measurements per image
    if args.type=='carapace':
        high_error_df = df[abs(df['mae']) >3]
    else:
        high_error_df = df[abs(df['mae']) >10]
      # Filter for high errors
    high_error_counts_by_image = high_error_df.groupby('Label').size()

# Reindex both Series to ensure they have the same index
# This is critical for correct comparison
    total_measurements_by_image = total_measurements_by_image.reindex(all_image_labels, fill_value=0)
    high_error_counts_by_image = high_error_counts_by_image.reindex(all_image_labels, fill_value=0)

# Find images where all measurements have high errors (and have more than 1 measurement)
    image_100_error_rate = high_error_counts_by_image[
    (high_error_counts_by_image == total_measurements_by_image) & 
    (total_measurements_by_image > 1)
].index.tolist()

# Flag measurements from these images
    df['flag_all_high_error_rate_image'] = df['Label'].isin(image_100_error_rate)

# Find images with multiple high errors
    images_with_multiple_high_errors = high_error_counts_by_image[high_error_counts_by_image > 1].index.tolist()

# Flag measurements from images with multiple high errors
    df['flag_image_multiple_errors'] = df['Label'].isin(images_with_multiple_high_errors)



    flag_columns = ['flag_all_high_pixel_error', 'flag_all_high_gt_expert_error', 
                        'flag_all_high_error_rate_image', 'flag_image_multiple_errors','high_error']
    


# Create a new comprehensive box plot with mutually exclusive categories
    exclusive_flags_fig = go.Figure()

# Define priorities based on percentage thresholds
# New approach: Add columns with the actual percentage values for each flag

    df['pred_gt_diff_pct'] = (df['pred_pixel_gt_diff'] / df['pred_Distance_pixels'] * 100)

# Set the pose value to a comparable scale (0-100)
    if 'pose_eval' in df.columns:
        df['pose_pct'] = (1 - df['pose_eval']) * 100  # Convert to percent error (higher = worse)
    elif 'pose_eval_iou' in df.columns:
        df['pose_pct'] = (1 - df['pose_eval_iou']) * 100
    else:
        df['pose_pct'] = 0

# Now determine the primary flag based on the highest percentage
    def get_primary_flag_by_pct(row):
        # First check if we have any errors at all

        # if row['flag_justified']:
        #     return 'Justified'
        if not row['high_error']:
            return 'No Flags'
    
        # Now we know high_error is True, so no need to check it agai
        elif row['flag_pred_gt_diff'] and row['pred_gt_diff_pct']>row['avg_gt_expert_error_pct']:
            return 'Prediction-GT pixel diff over 10%'
        
        elif row['flag_low_pose_eval']:
            return 'Pose error >5%'

        
        elif row['flag_high_avg_pixel_error'] and not row['flag_high_avg_gt_expert_error']:
            return 'pred pixel error >10%'        
        
       
        
        elif row['flag_high_avg_gt_expert_error'] :
            return 'GT-Expert pixel diff over >10%'
        





        
        # elif row['flag_high_avg_gt_expert_error'] and row['flag_avg_scale_error']:
        #     return 'All GT-Expert error >10% and Scale error >10%'

        elif row['flag_avg_scale_error'] and row['flag_image_multiple_errors']:
            return 'Scale error >10% and multiple errors'

        elif row['flag_avg_scale_error'] and row['flag_high_diff_min_max_length']:
            return 'Scale error >10% and min-max length diff >5'


        elif row['flag_avg_scale_error']or abs(row['mean_annotation_length']-row['avg_length'])<10:
                    return 'Scale error >10%'

       
        else:   
            return 'Unclassified Error'  # More descriptive than 'strange'

# Assign each measurement to exactly one category based on highest percentage
    df['assigned_category'] = df.apply(get_primary_flag_by_pct, axis=1)

# Create a category for measurements with no flags
    # df.loc[df['assigned_category'].isna(), 'assigned_category'] = 'No Flags'

# The order for visualization still needs to be defined
    priority_names = [
    'Prediction-GT pixel diff over 10%',
        'GT-Expert pixel diff over >10%',
        'pred pixel error >10%',
        'Scale error >10%',
        'Pose error >5%',
        'Scale error >10% and multiple errors',
        'Scale error >10% and min-max length diff >5',
        'Unclassified Error',
        'Justified'
]
    

    df['min_error_pixels']= df[['Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].min(axis=1)


# # Print some statistics about our prioritized assignments
#     print("\nAssignments based on percentage values:")
#     for category in priority_names + ['No Flags']:
#         count = len(df[df['assigned_category'] == category])
#         print(f"{category}: {count} measurements")

# Store colors for visualization
    priority_colors = ['#9b59b6', '#2ecc71', '#f39c12', '#3498db', '#e74c3c', '#c0392b', '#8e44ad', '#27ae60', '#f1c40f', '#e67e22', '#34495e']
    df['flag_count'] = 0
    for flag in flag_columns:
        df['flag_count'] += df[flag].astype(int) 

    df['min_gt_diff']=0
    df['gt_diff_pct']=0
# Add a box plot for each exclusive category



#for each pond type
    for pond_type in df['Pond_Type'].unique():
        # Create a new figure for each pond type
        pond_fig = go.Figure()
        
        pond_df = df[df['Pond_Type'] == pond_type]
        

        categories = [cat for cat in priority_names] + ['No Flags']
        for i, category in enumerate(categories):
            cat_df = pond_df[pond_df['assigned_category'] == category]
            
            if len(cat_df) > 0:  # Only add if there are measurements in this category
                color = priority_colors[i] if i < len(priority_colors) else 'gray'
                
                pond_fig.add_trace(go.Box(
                    y=cat_df['mae'],  # Use cat_df instead of pond_df
                    name=category,
                    boxmean=True,
                    marker_color=color,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.5,
                    width=0.6,
                    quartilemethod="linear",
                    hovertemplate=
                        f"<b>{category}</b><br>" +
                        "<b>ID:</b> %{customdata[0]}<br>" +
                        "<b>Image:</b> %{customdata[1]}<br>" +
                        "<b>Pond Type:</b> %{customdata[9]}<br>" +
                        "<b>Error:</b> %{y:.1f}mm<br>" +
                        "<b>GT Diff:</b> %{customdata[2]:.1f}px (%{customdata[4]:.1f}%)<br>" +
                        "<b>Pred-GT Diff:</b> %{customdata[5]:.1f}px (%{customdata[6]:.1f}%)<br>" +
                        "<b>Pred Diff:</b> %{customdata[7]:.1f}px (%{customdata[8]:.1f}%)<br>" +
                        "<b>Flag Count:</b> %{customdata[3]}<br>" +
                        "<b>best pixels</b> %{customdata[10]:.1f}px<br>" +
                        "<b>pred pixels</b> %{customdata[11]:.1f}px<br>" +
                        "<b>pixel diff 1</b> %{customdata[12]:.1f}%<br>" +
                        "<b>pixel diff 2</b> %{customdata[13]:.1f}%<br>" +
                        "<b>pixel diff 3</b> %{customdata[14]:.1f}%<br>" +
                        "<b>gt expert diff 1</b> %{customdata[15]:.1f}%<br>" +
                        "<b>gt expert diff 2</b> %{customdata[16]:.1f}%<br>" +
                        "<b>gt expert diff 3</b> %{customdata[17]:.1f}%<br>" +
                        "<b>pose error</b> %{customdata[18]:.1f}%<br>" +
                        "<b>avg scale error</b> %{customdata[19]:.1f}%<br>" +
                        "<b>avg gt expert error</b> %{customdata[20]:.1f}%<br>" +
                        "<b>Length_fov(mm)</b> %{customdata[21]:.1f}<br>" +
                        "<b>avg length</b> %{customdata[22]:.1f}<br>" +
                        "<b>avg pixel error</b> %{customdata[23]:.1f}%<br>" +
                        "<b>mean annotation length</b> %{customdata[24]:.1f}<br>",
                    customdata=cat_df[['PrawnID', 'Label', 'min_gt_diff', 'flag_count', 
                                    'gt_diff_pct', 'pred_pixel_gt_diff', 'pred_gt_diff_pct',
                                    'min_error_pixels', 'min_mape_pixels','Pond_Type','best_length_pixels',
                                    'pred_Distance_pixels','pixel_diff_pct_1','pixel_diff_pct_2',
                                    'pixel_diff_pct_3','gt_expert_diff_pct_1','gt_expert_diff_pct_2',
                                    'gt_expert_diff_pct_3','pose_pct','avg_scale_error','avg_gt_expert_error_pct','Length_fov(mm)','avg_length','avg_pixel_error_pct','mean_annotation_length']].values
                ))
        if args.type=='carapace':
            y0=3
        else:
            y0=10
        # Add horizontal line at 15% error
        pond_fig.add_shape(
            type='line',
            x0=-0.5, x1=len(categories) - 0.5,
            y0=y0, y1=y0,
            line=dict(color='red', dash='dash')
        )
        
        # Update layout
        pond_fig.update_layout(
            title=f'Error Distribution by Categories for {pond_type}',
            yaxis_title='Min MPE (%)',
            height=800, width=2000,
            boxmode='group',
            yaxis=dict(
                range=[-50, max(50, df['mpe'].max() * 1.1)]
            ),
            margin=dict(l=50, r=50, t=80, b=120)
        )

        # Add counts to the box plot names
        for i, trace in enumerate(pond_fig.data):
            category = trace.name
            count = len(pond_df[pond_df['assigned_category'] == category])
            trace.name = f"{category} (n={count})"

        os.makedirs(f'graphs/{args.type}', exist_ok=True)
        pond_fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html')


        # Create scatter plot with error bands
        scatter_fig = go.Figure()

        # Add diagonal reference line (y=x)
        scatter_fig.add_trace(go.Scatter(
            x=[min(pond_df['Length_fov(mm)']), max(pond_df['Length_fov(mm)'])],
            y=[min(pond_df['Length_fov(mm)']), max(pond_df['Length_fov(mm)'])],
            mode='lines',
            name='Perfect prediction (y=x)',
            line=dict(color='black', dash='dash')
        ))

        # Add ±3mm error bands
        scatter_fig.add_trace(go.Scatter(
            x=[min(pond_df['Length_fov(mm)']), max(pond_df['Length_fov(mm)'])],
            y=[min(pond_df['Length_fov(mm)']) + 3, max(pond_df['Length_fov(mm)']) + 3],
            mode='lines',
            name='+3mm',
            line=dict(color='red', dash='dot')
        ))

        scatter_fig.add_trace(go.Scatter(
            x=[min(pond_df['Length_fov(mm)']), max(pond_df['Length_fov(mm)'])],
            y=[min(pond_df['Length_fov(mm)']) - 3, max(pond_df['Length_fov(mm)']) - 3],
            mode='lines',
            name='-3mm',
            line=dict(color='red', dash='dot')
        ))

        # Add scatter plot
        scatter_fig.add_trace(go.Scatter(
            x=pond_df['Length_fov(mm)'],
            y=pond_df['mean_length'],
            mode='markers',
            name='Data Points',
            hovertemplate=
                "<b>ID:</b> %{customdata[0]}<br>" +
                "<b>Image:</b> %{customdata[1]}<br>" +
                "<b>Predicted Length:</b> %{x:.1f}mm<br>" +
                "<b>Mean Expert Length:</b> %{y:.1f}mm<br>" +
                "<b>Absolute Difference:</b> %{customdata[2]:.1f}mm<br>",
            customdata=np.column_stack((
                pond_df['PrawnID'],
                pond_df['Label'],
                abs(pond_df['mean_length'] - pond_df['Length_fov(mm)'])
            ))
        ))

        # Update layout
        scatter_fig.update_layout(
            title=f'Predicted vs Mean Expert Length for {pond_type}',
            xaxis_title='Predicted Length (mm)',
            yaxis_title='Mean Expert Length (mm)',
            height=800,
            width=800,
            showlegend=True,
            xaxis=dict(
                range=[min(pond_df['Length_fov(mm)']), max(pond_df['Length_fov(mm)'])],
                scaleanchor='y',
                scaleratio=1,
            ),
            yaxis=dict(
                range=[min(pond_df['Length_fov(mm)']), max(pond_df['Length_fov(mm)'])],
            )
        )

        # Save the scatter plot
        scatter_fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/scatter_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html')

        # Create pixel-to-pixel scatter plot
        pixel_scatter_fig = go.Figure()
        
        # Calculate mean expert pixels
        pond_df['mean_expert_pixels'] = pond_df[['Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].mean(axis=1)
        
        # Calculate linear regression for pixels
        # slope_px, intercept_px, r_value_px, p_value_px, std_err_px = stats.linregress(pond_df['pred_Distance_pixels'], pond_df['mean_expert_pixels'])
        # r_squared_px = r_value_px**2
        
        # Find min and max for pixel values with padding
        min_val_px = min(pond_df['pred_Distance_pixels'].min(), pond_df['mean_expert_pixels'].min()) * 0.8
        max_val_px = max(pond_df['pred_Distance_pixels'].max(), pond_df['mean_expert_pixels'].max()) * 1.2
        
        # Add diagonal reference line (y=x)
        pixel_scatter_fig.add_trace(go.Scatter(
            x=[min_val_px, max_val_px],
            y=[min_val_px, max_val_px],
            mode='lines',
            name='Perfect prediction (y=x)',
            line=dict(color='black', dash='dash')
        ))
        
        # Add scatter plot for pixels
        pixel_scatter_fig.add_trace(go.Scatter(
            x=pond_df['pred_Distance_pixels'],
            y=pond_df['mean_expert_pixels'],
            mode='markers',
            name='Data Points',
            hovertemplate=
                "<b>ID:</b> %{customdata[0]}<br>" +
                "<b>Image:</b> %{customdata[1]}<br>" +
                "<b>Predicted Pixels:</b> %{x:.1f}px<br>" +
                "<b>Mean Expert Pixels:</b> %{y:.1f}px<br>" +
                "<b>Expert 1 Pixels:</b> %{customdata[2]:.1f}px<br>" +
                "<b>Expert 2 Pixels:</b> %{customdata[3]:.1f}px<br>" +
                "<b>Expert 3 Pixels:</b> %{customdata[4]:.1f}px<br>",
            customdata=pond_df[['PrawnID', 'Label', 'Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].values
        ))
        
        # Update layout for pixel scatter plot
        pixel_scatter_fig.update_layout(
            title=f'Predicted vs Mean Expert Pixels for {pond_type}',
            xaxis_title='Predicted Length (pixels)',
            yaxis_title='Mean Expert Length (pixels)',
            height=800,
            width=800,
            showlegend=True,
            xaxis=dict(
                range=[min_val_px, max_val_px],
                scaleanchor='y',
                scaleratio=1,
            ),
            yaxis=dict(
                range=[min_val_px, max_val_px]
            )
        )
        
        # Save the pixel scatter plot
        pixel_scatter_fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/scatter_pixels_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html')



    #scatter plot of expert mean and anonotation length 
    scatter_fig = go.Figure()
    pond_df['mean_expert_pixels'] = pond_df[['Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].mean(axis=1)
    #anotation
    #ground trtuh annotation length

    # Add diagonal reference line (y=x)
    scatter_fig.add_trace(go.Scatter(
        x=pond_df['mean_expert_pixels'],
        y=pond_df['Length_ground_truth_annotation_pixels'],
        mode='markers',
        name='Data Points',
        hovertemplate=
            "<b>ID:</b> %{customdata[0]}<br>" +
            "<b>Image:</b> %{customdata[1]}<br>" +
            "<b>Mean Expert Pixels:</b> %{x:.1f}px<br>" +
            "<b>Ground Truth Annotation Length:</b> %{y:.1f}px<br>" +
            "<b>Absolute Difference:</b> %{customdata[2]:.1f}px<br>",
        customdata=np.column_stack((
            pond_df['PrawnID'],
            pond_df['Label'],
            abs(pond_df['mean_expert_pixels'] - pond_df['Length_ground_truth_annotation_pixels'])
        ))
    ))
    #add line of x=y
    scatter_fig.add_trace(go.Scatter(
        x=[min(pond_df['mean_expert_pixels']), max(pond_df['mean_expert_pixels'])],
        y=[min(pond_df['Length_ground_truth_annotation_pixels']), max(pond_df['Length_ground_truth_annotation_pixels'])],
        mode='lines',
        name='Perfect prediction (y=x)',
        line=dict(color='black', dash='dash')
    ))
    # Update layout
    scatter_fig.update_layout(
        title='Mean Expert Pixels vs Ground Truth Annotation Length',
        xaxis_title='Mean Expert Pixels',
        yaxis_title='Ground Truth Annotation Length (pixels)',
        height=800,
        width=800,
        showlegend=True,
        xaxis=dict(
            range=[min(pond_df['mean_expert_pixels']), max(pond_df['mean_expert_pixels'])],
            scaleanchor='y',
            scaleratio=1,
        ),
        yaxis=dict( 
            range=[min(pond_df['Length_ground_truth_annotation_pixels']), max(pond_df['Length_ground_truth_annotation_pixels'])],
        )
    )

    # Save the scatter plot
    scatter_fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/scatter_expert_mean_annotation_length_{args.type}_{args.weights_type}_{args.error_size}.html')
    


    #create a 3d scatter plot of avg scale error, avg pixel error, mpe
    # Create 3D scatter plot for each pond type
    for pond_type in df['Pond_Type'].unique():
        pond_df = df[df['Pond_Type'] == pond_type]
        
        # Calculate correlation coefficients
        corr_scale_mpe = pond_df['avg_scale_error'].corr(pond_df['mpe'])
        corr_pixel_mpe = pond_df['avg_pixel_error_pct'].corr(pond_df['mpe'])
        corr_scale_pixel = pond_df['avg_scale_error'].corr(pond_df['avg_pixel_error_pct'])


        pond_df['combined_multiple'] = pond_df['avg_scale_error'] * pond_df['avg_pixel_error_pct']

        corr_combined_multiple = pond_df['combined_multiple'].corr(pond_df['mpe']) 
        
        fig = go.Figure(data=[go.Scatter3d(
            x=pond_df['avg_scale_error'],
            y=pond_df['avg_pixel_error_pct'], 
            z=pond_df['mpe'],
            mode='markers',
            marker=dict(
                size=5,
                color=pond_df['combined_multiple'],  # Color points by MPE
                colorscale='Viridis',
                showscale=True
            ),
            name='Data Points',
            hovertemplate=
                "<b>ID:</b> %{customdata[0]}<br>" +
                "<b>Image:</b> %{customdata[1]}<br>" +
                "<b>Avg Scale Error:</b> %{x:.1f}%<br>" +
                "<b>Avg Pixel Error:</b> %{y:.1f}%<br>" +
                "<b>MPE:</b> %{z:.1f}%<br>",
            customdata=pond_df[['PrawnID', 'Label']].values
        )])

        # Update layout with correlation information
        fig.update_layout(
            title=f'3D Scatter Plot with Correlations for {pond_type}<br>' +
                  f'Scale-MPE Correlation: {corr_scale_mpe:.2f}<br>' +
                  f'Pixel-MPE Correlation: {corr_pixel_mpe:.2f}<br>' +
                  f'Scale-Pixel Correlation: {corr_scale_pixel:.2f}<br>' +
                  f'Combined Multiple-MPE Correlation: {corr_combined_multiple:.2f}',
            scene=dict(
                xaxis_title='Avg Scale Error (%)',
                yaxis_title='Avg Pixel Error (%)', 
                zaxis_title='MPE (%)'
            ),
            height=800,
            width=1000
        )

        # Save the 3D scatter plot for this pond type
        fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/3d_scatter_avg_scale_error_avg_pixel_error_mpe_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html')
    
        #create a 2d scatter plot of pixel error mpe
        fig = go.Figure()
        # Calculate R-squared
        slope, intercept = np.polyfit(pond_df['avg_pixel_error_pct'], pond_df['mpe'], 1)
        r_squared = r2_score(pond_df['mpe'], slope * pond_df['avg_pixel_error_pct'] + intercept)
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=pond_df['avg_pixel_error_pct'],
            y=pond_df['mpe'],
            mode='markers',
            name=f'Data Points (R² = {r_squared:.3f})',
            hovertemplate=
                "<b>ID:</b> %{customdata[0]}<br>" +
                "<b>Image:</b> %{customdata[1]}<br>" +
                "<b>Pixel Error:</b> %{x:.1f}%<br>" +
                "<b>MPE:</b> %{y:.1f}%<br>",
            customdata=pond_df[['PrawnID', 'Label']].values
        ))
        # Add regression line
        fig.add_trace(go.Scatter(
            x=pond_df['avg_pixel_error_pct'],
            y=slope * pond_df['avg_pixel_error_pct'] + intercept,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', dash='dash')
        ))
        # Update layout
        fig.update_layout(
            title='Pixel Error vs MPE',
            xaxis_title='Pixel Error (%)',
            yaxis_title='MPE (%)',
            height=800,
            width=1000
        )
        # Save the scatter plot
        fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/scatter_pixel_error_mpe_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html')
        
        

        #create a 2d scatter plot of scale error mpe
        fig = go.Figure()
        # Calculate R-squared
        slope, intercept = np.polyfit(pond_df['avg_scale_error'], pond_df['mpe'], 1)
        r_squared = r2_score(pond_df['mpe'], slope * pond_df['avg_scale_error'] + intercept)
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=pond_df['avg_scale_error'],
            y=pond_df['mpe'],
            mode='markers',
            name=f'Data Points (R² = {r_squared:.3f})',
            hovertemplate=
                "<b>ID:</b> %{customdata[0]}<br>" +
                "<b>Image:</b> %{customdata[1]}<br>" +
                "<b>Scale Error:</b> %{x:.1f}%<br>" +
                "<b>MPE:</b> %{y:.1f}%<br>",
            customdata=pond_df[['PrawnID', 'Label']].values
        ))
        # Add regression line
        fig.add_trace(go.Scatter(
            x=pond_df['avg_scale_error'],
            y=slope * pond_df['avg_scale_error'] + intercept,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', dash='dash')
        ))
        # Update layout
        fig.update_layout(
            title='Scale Error vs MPE',
            xaxis_title='Scale Error (%)',
            yaxis_title='MPE (%)',
            height=800,
            width=1000
        )
        # Save the scatter plot
        fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/scatter_scale_error_mpe_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html')
        
        
        
        #scatter plot of pixel+scael mpe
        pond_df['added_multiple'] = pond_df['avg_scale_error'] + pond_df['avg_pixel_error_pct']

        # Calculate R-squared
        slope, intercept = np.polyfit(pond_df['added_multiple'], pond_df['mpe'], 1)
        r_squared = r2_score(pond_df['mpe'], slope * pond_df['added_multiple'] + intercept)
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=pond_df['added_multiple'],
            y=pond_df['mpe'],
            mode='markers',
            name=f'Data Points (R² = {r_squared:.3f})',
            hovertemplate=
                "<b>ID:</b> %{customdata[0]}<br>" +
                "<b>Image:</b> %{customdata[1]}<br>" +
                "<b>Added Multiple:</b> %{x:.1f}<br>" +
                "<b>MPE:</b> %{y:.1f}%<br>",
            customdata=pond_df[['PrawnID', 'Label']].values
        ))
        # Add regression line
        fig.add_trace(go.Scatter(
            x=pond_df['added_multiple'],
            y=slope * pond_df['added_multiple'] + intercept,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', dash='dash')
        ))
        # Update layout
        fig.update_layout(
            title='Pixel+Scale Error vs MPE',
            xaxis_title='Pixel+Scale Error (%)',
            yaxis_title='MPE (%)',
            height=800,
            width=1000
        )
        # Save the scatter plot
        fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/scatter_pixel_scale_error_mpe_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html')
        
        
    #measument uncertainty distribution across dataset
    #plot the distribution of the measurement uncertainty







    for pond_type in df['Pond_Type'].unique():
        df_pond_type = df[df['Pond_Type'] == pond_type]

        #std of length_1, length_2, length_3
        df_pond_type.loc[:, 'std'] = df_pond_type[['Length_1','Length_2','Length_3']].std(axis=1)
        df_pond_type.loc[:, 'max'] = df_pond_type[['Length_1','Length_2','Length_3']].max(axis=1)
        df_pond_type.loc[:, 'min'] = df_pond_type[['Length_1','Length_2','Length_3']].min(axis=1)
        df_pond_type.loc[:, 'mean'] = df_pond_type[['Length_1','Length_2','Length_3']].mean(axis=1)

        #plot the distribution of the measurement uncertainty
        ranges = df_pond_type['max'] - df_pond_type['min']
        means = df_pond_type['mean']

    # Ensure x and y arrays have the same length
        x_values = range(1, len(means) + 1)
        plt.errorbar(x_values, means, yerr=ranges/2, fmt='o', ecolor='orange', capsize=3)
        plt.xlabel("Prawn Index")
        plt.ylabel("Measurement")
        plt.title(f"Measurement Uncertainty (Mean ± Half Range) for {pond_type}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/measurement_uncertainty_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.png')
        # plt.show()


    #the mean of ranges sns kde plot
        sns.kdeplot(ranges, fill=True)
        plt.xlabel("Range")
        plt.ylabel("Frequency")
        plt.title(f"Range Distribution for {pond_type}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/range_distribution_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.png')
        # plt.show()



        df_pond_type.loc[:, 'std_pixels'] = df_pond_type[['Length_1_pixels','Length_2_pixels','Length_3_pixels']].std(axis=1)
        df_pond_type.loc[:, 'max_pixels'] = df_pond_type[['Length_1_pixels','Length_2_pixels','Length_3_pixels']].max(axis=1)
        df_pond_type.loc[:, 'min_pixels'] = df_pond_type[['Length_1_pixels','Length_2_pixels','Length_3_pixels']].min(axis=1)
        df_pond_type.loc[:, 'mean_pixels'] = df_pond_type[['Length_1_pixels','Length_2_pixels','Length_3_pixels']].mean(axis=1)

        #plot the distribution of the measurement uncertainty
        ranges = df_pond_type['max_pixels'] - df_pond_type['min_pixels']
        means = df_pond_type['mean_pixels']

        # Ensure x and y arrays have the same length
        x_values = range(1, len(means) + 1)
        plt.errorbar(x_values, means, yerr=ranges/2, fmt='o', ecolor='orange', capsize=3)
        plt.xlabel("Prawn Index")
        plt.ylabel("Measurement")
        plt.title(f"Pixel Measurement Uncertainty (Mean ± Half Range) for {pond_type}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/pixel_measurement_uncertainty_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.png')
        # plt.show()


        #the mean of ranges sns kde plot
        sns.kdeplot(ranges, fill=True)
        plt.xlabel("Range")
        plt.ylabel("Frequency")
        plt.title(f"Pixel Range Distribution for {pond_type}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/pixel_range_distribution_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.png')
        # plt.show()


      

        #avg scale
        df_pond_type.loc[:, 'mean_scale'] = df_pond_type[['scale_normalized_1', 'scale_normalized_2', 'scale_normalized_3']].mean(axis=1)


        df_pond_type.loc[:, 'std_scale'] = df_pond_type[['scale_normalized_1','scale_normalized_2','scale_normalized_3']].std(axis=1)
        df_pond_type.loc[:, 'max_scale'] = df_pond_type[['scale_normalized_1','scale_normalized_2','scale_normalized_3']].max(axis=1)
        df_pond_type.loc[:, 'min_scale'] = df_pond_type[['scale_normalized_1','scale_normalized_2','scale_normalized_3']].min(axis=1)
        df_pond_type.loc[:, 'mean_scale'] = df_pond_type[['scale_normalized_1','scale_normalized_2','scale_normalized_3']].mean(axis=1)

        #plot the distribution of the measurement uncertainty
        ranges = df_pond_type['max_scale'] - df_pond_type['min_scale']
        means = df_pond_type['mean_scale']

        # Ensure x and y arrays have the same length
        x_values = range(1, len(means) + 1)
        plt.errorbar(x_values, means, yerr=ranges/2, fmt='o', ecolor='orange', capsize=3)
        plt.xlabel("Prawn Index")
        plt.ylabel("Measurement")
        plt.title(f"Scale Measurement Uncertainty (Mean ± Half Range) for {pond_type}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/scale_measurement_uncertainty_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.png')
        # plt.show()


        sns.kdeplot(ranges, fill=True)
        plt.xlabel("Range")
        plt.ylabel("Frequency")
        plt.title(f"Scale Range Distribution for {pond_type}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/scale_range_distribution_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.png')
        # plt.show()



        # kde plot of scale error
        sns.kdeplot(df_pond_type['avg_scale_diff'], fill=True)
        plt.xlabel("Scale Error")
        plt.ylabel("Frequency")
        plt.title(f"Scale Error Distribution for {pond_type}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/scale_error_distribution_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.png')
        # plt.show()


        # kde plot of pixel error
        sns.kdeplot(df_pond_type['avg_pixel_diff'], fill=True)
        plt.xlabel("Pixel Error")
        plt.ylabel("Frequency")
        plt.title(f"Pixel Error Distribution for {pond_type}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/pixel_error_distribution_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.png')
        # plt.show()


        #one 3d scatter plot of lengths, pixels, scale
        fig = px.scatter_3d(df_pond_type, x='mean', y='mean_pixels', z='mean_scale', color='Pond_Type')
        fig.update_layout(
            title='3D Scatter Plot of Lengths, Pixels, and Scale',
            scene=dict(
                xaxis_title='Length',
                yaxis_title='Pixels',
                zaxis_title='Scale'
            )
        )
        fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/results/analysis/graphs/{args.type}/length_pixels_scale_scatter_{args.type}_{args.weights_type}_{args.error_size}_{pond_type}.html')
        # fig.show()









    #     #add a box plot for each pond type
#     annotation_length_fig.add_shape(
#     type='line',
#     x0=-0.5, x1=len(categories) - 0.5,
#         y0=y0, y1=y0,
#     line=dict(color='red', dash='dash')
# )   

#     #update layout
#     annotation_length_fig.update_layout(
#     title='Annotation Length Distribution',
#     yaxis_title='Annotation Length (%)',
#     height=800, width=2000,
# )

#     #add counts to the box plot names
#     for i, trace in enumerate(annotation_length_fig.data):
#         category = trace.name
#         count = len(df[df['Pond_Type'] == category])
#         trace.name = f"{category} (n={count})"  

#     os.makedirs('graphs', exist_ok=True)
#     annotation_length_fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/analysis/graphs/annotation_length_fig_{args.type}_{args.weights_type}_{args.error_size}.html')











    """
    statistics:
    -----------
    mae: mean absolute error without flags
    mae by pond type: mean absolute error by pond type without flags
    mape: mean absolute percentage error without flags
    mape by pond type: mean absolute percentage error by pond type without flags
    mpe: mean percentage error without flags
    mpe by pond type: mean percentage error by pond type without flags
    rmse: root mean square error without flags
    rmse by pond type: root mean square error by pond type without flags
    r2: r-squared without flags

    mae_flags: mean absolute error with flags
    mape_flags: mean absolute percentage error with flags
    mpe_flags: mean percentage error with flags
    rmse_flags: root mean square error with flags
    r2_flags: r-squared with flags

    """

    mae_flags = df[df['assigned_category'] == 'No Flags']['mae'].mean()
    mape_flags = df[df['assigned_category'] == 'No Flags']['mpe'].mean()
    mae_by_pond_type = df[df['assigned_category'] == 'No Flags'].groupby('Pond_Type')['mae'].mean()
    mape_by_pond_type = df[df['assigned_category'] == 'No Flags'].groupby('Pond_Type')['mpe'].mean()
    

    std_mae_flags = df[df['assigned_category'] == 'No Flags']['mae'].std()
    std_mape_flags = df[df['assigned_category'] == 'No Flags']['mpe'].std()

    std_mae_by_pond_type = df[df['assigned_category'] == 'No Flags'].groupby('Pond_Type')['mae'].std()
    std_mape_by_pond_type = df[df['assigned_category'] == 'No Flags'].groupby('Pond_Type')['mpe'].std()



    mae_with_flags = df[df['assigned_category'] != 'No Flags']['mae'].mean()
    mape_with_flags = df[df['assigned_category'] != 'No Flags']['mpe'].mean()
    mae_by_pond_type_with_flags = df[df['assigned_category'] != 'No Flags'].groupby('Pond_Type')['mae'].mean()
    mape_by_pond_type_with_flags = df[df['assigned_category'] != 'No Flags'].groupby('Pond_Type')['mpe'].mean()

    std_mae_with_flags = df[df['assigned_category'] != 'No Flags']['mae'].std()
    std_mape_with_flags = df[df['assigned_category'] != 'No Flags']['mpe'].std()

    std_mae_by_pond_type_with_flags = df[df['assigned_category'] != 'No Flags'].groupby('Pond_Type')['mae'].std()
    std_mape_by_pond_type_with_flags = df[df['assigned_category'] != 'No Flags'].groupby('Pond_Type')['mpe'].std()



    # # Print MAE and MPE by category
    # print("\n=== MAE and MPE by Category ===")
    # print(f"{'Category':<30} {'MAE Without Category':>25} {'MAE With Only Category':>25} {'MPE Without Category':>25} {'MPE With Only Category':>25}")
    # print("-" * 130)


    
    for category in categories:
        # Calculate MAE and MPE without the current category
        mae_without_category = df[df['assigned_category'] != category]['mae'].mean()
        mape_without_category = df[df['assigned_category'] != category]['mpe'].mean()


        # Calculate MAE and MPE with only the current category
        mae_with_only_category = df[df['assigned_category'] == category]['mae'].mean()
        mape_with_only_category = df[df['assigned_category'] == category]['mpe'].mean()
        

        std_mae_without_category = df[df['assigned_category'] != category]['mae'].std()
        std_mape_without_category = df[df['assigned_category'] != category]['mpe'].std()

        std_mae_with_only_category = df[df['assigned_category'] == category]['mae'].std()
        std_mape_with_only_category = df[df['assigned_category'] == category]['mpe'].std()


        # print(f"{category:<30} {mae_without_category:>25.2f}±{std_mae_without_category:.2f} {mae_with_only_category:>25.2f}±{std_mae_with_only_category:.2f} {mape_without_category:>25.2f}±{std_mape_without_category:.2f} {mape_with_only_category:>25.2f}±{std_mape_with_only_category:.2f} {std_mae_flags:>25.2f} {std_mape_flags:>25.2f} ")







    mae_high_error_rate_image_and_gt_expert_error_and_high_pixel_error = df[
        (df['flag_all_high_error_rate_image'] == 1) |
        (df['flag_all_high_gt_expert_error'] == 1) |
        (df['flag_all_high_pixel_error'] == 1) |
        (df['flag_image_multiple_errors'] == 1)
    ]['pixel_pct'].mean()

    mape_high_error_rate_image_and_gt_expert_error_and_high_pixel_error = df['pixel_pct'].mean()

    # print(f"{'High error rate image and gt expert error and high pixel error':<30} {mae_high_error_rate_image_and_gt_expert_error_and_high_pixel_error:>25.2f} {mape_high_error_rate_image_and_gt_expert_error_and_high_pixel_error:>25.2f}")
    
    
    

    # for category in categories:
    #     for pond_type in df['Pond_Type'].unique():
    #         print(f"\n=== {category} for {pond_type} ===")
    #         print(f"{'Metric':<30} {'Without Category':>20} {'With Only Category':>20}")
    #         print("-" * 70)
            
    #         # Calculate statistics for the current category and pond type
    #         mae_without_category = df[(df['assigned_category'] != category) & (df['Pond_Type'] == pond_type)]['mae'].mean()
    #         mape_without_category = df[(df['assigned_category'] != category) & (df['Pond_Type'] == pond_type)]['mpe'].mean()
            
    #         mae_with_only_category = df[(df['assigned_category'] == category) & (df['Pond_Type'] == pond_type)]['mae'].mean()
    #         mape_with_only_category = df[(df['assigned_category'] == category) & (df['Pond_Type'] == pond_type)]['mpe'].mean()

    #         std_mae_without_category = df[(df['assigned_category'] != category) & (df['Pond_Type'] == pond_type)]['mae'].std()
    #         std_mape_without_category = df[(df['assigned_category'] != category) & (df['Pond_Type'] == pond_type)]['mpe'].std()

    #         std_mae_with_only_category = df[(df['assigned_category'] == category) & (df['Pond_Type'] == pond_type)]['mae'].std()
    #         std_mape_with_only_category = df[(df['assigned_category'] == category) & (df['Pond_Type'] == pond_type)]['mpe'].std()   

    #         print(f"{category:<30} {mae_without_category:>25.2f} + {std_mae_without_category:>25.2f} {mae_with_only_category:>25.2f} + {std_mae_with_only_category:>25.2f} {mape_without_category:>25.2f} + {std_mape_without_category:>25.2f} {mape_with_only_category:>25.2f} + {std_mape_with_only_category:>25.2f}")

            
        





    # # Existing overall statistics table
    # print("\n=== Overall Statistics ===")
    # print(f"{'Metric':<30} {'Without Flags':>15} {'With Flags':>15}")
    # print("-" * 60)
    # print(f"{'MAE (Mean Absolute Error)':<30} {mae_flags:>15.2f}±{std_mae_flags:>25.2f} {mae_with_flags:>15.2f}±{std_mae_with_flags:>25.2f}")
    # print(f"{'MAPE (Mean Absolute % Error)':<30} {mape_flags:>15.2f}±{std_mape_flags:>25.2f} {mape_with_flags:>15.2f}±{std_mape_with_flags:>25.2f}")



    

        #heatmap-style visualization showing how pixel and scale errors affect MPE    
    plt.figure(figsize=(10, 8))

    # KDE contour filled by MPE-weighted density
    sns.kdeplot(
        x=df['avg_pixel_error_pct'], 
        y=df['avg_scale_error'], 
        weights=df['mpe'],
        cmap="viridis", 
        fill=True, 
        thresh=0.05,
        levels=100
    )

    # Scatter points colored by MPE
    scatter = plt.scatter(
        df['avg_pixel_error_pct'], 
        df['avg_scale_error'], 
        c=df['mpe'], 
        cmap='viridis', 
        s=60
    )

    # Colorbar to explain MPE color intensity
    plt.colorbar(scatter, label='MPE (%)')

    # Labels and formatting
    plt.xlabel('Avg Pixel Error (%)')
    plt.ylabel('Avg Scale Error (%)')
    plt.title('Combined Influence of Pixel & Scale Errors on MPE')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()






# === Create Heatmap ===
    plt.figure(figsize=(12, 9))
    sns.kdeplot(
        x=df['avg_pixel_error_pct'], 
        y=df['avg_scale_error'], 
        weights=df['mpe'],
        cmap="viridis", 
        fill=True, 
        thresh=0.05,
        levels=100
    )

    # Overlay scatter points
    scatter = plt.scatter(
        df['avg_pixel_error_pct'], 
        df['avg_scale_error'], 
        c=df['mpe'], 
        cmap='viridis', 
        edgecolor='black',
        s=60
    )

    cbar = plt.colorbar(scatter, label='MPE (%)')
    plt.xlabel('Avg Pixel Error (%)')
    plt.ylabel('Avg Scale Error (%)')
    plt.title('Combined Influence of Pixel & Scale Errors on MPE')

    # === Annotate Key Observations & Hypotheses ===
    plt.text(25, 22, 
            '⬆ High Pixel & Scale Error\n→ High MPE\n→ Equation Amplifies Both',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.text(8, 6, 
            '⬅ Dense Low-Error Region\n→ Model is Robust Here',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.text(28, 8, 
            '⬆ Wide spread along pixel axis\n→ Pose Estimation Bottleneck?',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.text(10, 22, 
            '⬅ Tight spread in scale error\n→ Distance/Scale more stable',
            fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.grid(True)
    plt.tight_layout()
    # plt.show()

    df['mean_pixel_length'] = df[['Length_1_pixels','Length_2_pixels','Length_3_pixels']].mean(axis=1)
    
    # Calculate trend line and R squared for pixel lengths
    z1 = np.polyfit(df['mean_pixel_length'], df['pred_Distance_pixels'], 1)
    p1 = np.poly1d(z1)
    r_squared1 = np.corrcoef(df['mean_pixel_length'], df['pred_Distance_pixels'])[0,1]**2
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mean_pixel_length', y='pred_Distance_pixels')
    plt.plot(df['mean_pixel_length'], p1(df['mean_pixel_length']), "r--", alpha=0.8)
    plt.xlabel('Mean Expert Pixel Length (mm)')
    plt.ylabel('Pixel Length (mm)')
    plt.title(f'Mean Expert Pixel Length vs Pixel Length (R² = {r_squared1:.3f})')
    # plt.show()

    df['mean_length'] = df[['Length_1','Length_2','Length_3']].mean(axis=1)
    
    # Calculate trend line and R squared for lengths
    z2 = np.polyfit(df['Length_fov(mm)'], df['mean_length'], 1)
    p2 = np.poly1d(z2)
    r_squared2 = np.corrcoef(df['Length_fov(mm)'], df['mean_length'])[0,1]**2
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Length_fov(mm)', y='mean_length')
    plt.plot(df['Length_fov(mm)'], p2(df['Length_fov(mm)']), "r--", alpha=0.8)
    plt.xlabel('Length FOV (mm)')
    plt.ylabel('Mean Length (mm)')
    plt.title(f'Length FOV vs Mean Length (R² = {r_squared2:.3f})')
    # plt.show()

    df['mean_normalized_scale']=df[['scale_normalized_1','scale_normalized_2','scale_normalized_3']].mean(axis=1)
    # Calculate trend line and R squared for scales
    z3 = np.polyfit(df['mean_normalized_scale'], df['combined_scale'], 1)
    p3 = np.poly1d(z3)
    r_squared3 = np.corrcoef(df['mean_normalized_scale'], df['combined_scale'])[0,1]**2

    #scatter scale to scale
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mean_normalized_scale', y='combined_scale')
    plt.plot(df['mean_normalized_scale'], p3(df['mean_normalized_scale']), "r--", alpha=0.8)
    plt.xlabel('Scale')
    plt.ylabel('Combined Scale')
    plt.title(f'Scale vs combined scalex (R² = {r_squared3:.3f})')
    # plt.show()


    # # Print MAE by pond type table
    # print("\n=== MAE by Pond Type ===")
    # print(f"{'Pond Type':<20} {'Without Flags':>15} {'With Flags':>15}")
    # print("-" * 50)
    # for pond_type in mae_by_pond_type.index:
    #     print(f"{pond_type:<20} {mae_by_pond_type[pond_type]:>15.2f}±{std_mae_by_pond_type[pond_type]:>25.2f} {mae_by_pond_type_with_flags[pond_type]:>15.2f}±{std_mae_by_pond_type_with_flags[pond_type]:>25.2f}")

    # # Print MAPE by pond type table
    # print("\n=== MAPE by Pond Type ===")
    # print(f"{'Pond Type':<20} {'Without Flags':>15} {'With Flags':>15}")
    # print("-" * 50)
    # for pond_type in mape_by_pond_type.index:
    #     print(f"{pond_type:<20} {mape_by_pond_type[pond_type]:>15.2f}±{std_mape_by_pond_type[pond_type]:>25.2f} {mape_by_pond_type_with_flags[pond_type]:>15.2f}±{std_mape_by_pond_type_with_flags[pond_type]:>25.2f}")

    # # Print sample counts
    # print("\n=== Sample Counts ===")
    # print(f"{'Pond Type':<20} {'Without Flags':>15} {'With Flags':>15} {'Total':>15}")
    # print("-" * 65)
    # for pond_type in df['Pond_Type'].unique():
    #     without_flags = len(df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] == 'No Flags')])
    #     std_without_flags = df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] == 'No Flags')]['mae'].std()
    #     with_flags = len(df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] != 'No Flags')])
    #     std_with_flags = df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] != 'No Flags')]['mae'].std()
    #     total = without_flags + with_flags
    #     std_total = df[(df['Pond_Type'] == pond_type)]['mae'].std()
    #     print(f"{pond_type:<20} {without_flags:>15} + {std_without_flags:>25.2f} {with_flags:>15} + {std_with_flags:>25.2f} {total:>15} + {std_total:>25.2f}")

    # total_without_flags = len(df[df['assigned_category'] == 'No Flags'])
    # total_with_flags = len(df[df['assigned_category'] != 'No Flags'])
    # print("-" * 65)
    # print(f"{'Total':<20} {total_without_flags:>15} {total_with_flags:>15} {len(df):>15}")


    # #count statistics for each flag and pond type using primary flag using the assigned category
    # for category in categories:
    #     print(f"\n=== {category} Statistics ===")
    #     print(f"{'Pond Type':<20} {'Count':>15}")
    #     print("-" * 30)
        
    #     for pond_type in df['Pond_Type'].unique():
    #         count = len(df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] == category)])
    #         print(f"{pond_type:<20} {count:>15}")

        
    # #how much beneath 5% , how much between 5% and 10% , how much between 10% and 15% , how much above 15% 

    # #count how many are beneath 5%
    # count_beneath_5 = len(df[abs(df['mpe']) < 5])
    # print(f"{'Beneath 5%':<20} {count_beneath_5:>15}")

    # #count how many are between 5% and 10%
    # count_between_5_and_10 = len(df[(abs(df['mpe']) >= 5) & (abs(df['mpe']) <= 10)])
    # print(f"{'Between 5% and 10%':<20} {count_between_5_and_10:>15}")

    # #count how many are between 10% and 15%
    # count_between_10_and_15 = len(df[(abs(df['mpe']) >= 10) & (abs(df['mpe']) <= 15)])
    # print(f"{'Between 10% and 15%':<20} {count_between_10_and_15:>15}")

    # #count how many between 15% and 20%
    # count_between_15_and_20 = len(df[(abs(df['mpe']) >= 15) & (abs(df['mpe']) <= 20)])
    # print(f"{'Between 15% and 20%':<20} {count_between_15_and_20:>15}")

    # #count how many above 20%
    # count_above_20 = len(df[abs(df['mpe']) > 20])
    # print(f"{'Above 20%':<20} {count_above_20:>15}")

    #overall mpe
    #abs mpe
    df['mpe'] = abs(df['mpe'])
    overall_mpe = df['mpe'].mean()
    # print(f"{'Overall mpe':<20} {overall_mpe:>15}")


    #abs mpe annotation length
    df['mpe_annotation_length'] = abs(df['mpe_annotation_length'])

    # #count how many are beneath 5% annotation length
    # count_beneath_5_annotation_length = len(df[df['mpe_annotation_length'] < 5])
    # print(f"{'Beneath 5% annotation length':<20} {count_beneath_5_annotation_length:>15}")

    # #count how many are between 5% and 10% annotation length
    # count_between_5_and_10_annotation_length = len(df[(df['mpe_annotation_length'] >= 5) & (df['mpe_annotation_length'] <= 10)])
    # print(f"{'Between 5% and 10% annotation length':<20} {count_between_5_and_10_annotation_length:>15}")

    # #count how many are between 10% and 15% annotation length
    # count_between_10_and_15_annotation_length = len(df[(df['mpe_annotation_length'] >= 10) & (df['mpe_annotation_length'] <= 15)])
    # print(f"{'Between 10% and 15% annotation length':<20} {count_between_10_and_15_annotation_length:>15}")

    # #count how many are between 15% and 20% annotation length
    # count_between_15_and_20_annotation_length = len(df[(df['mpe_annotation_length'] >= 15) & (df['mpe_annotation_length'] <= 20)])
    # print(f"{'Between 15% and 20% annotation length':<20} {count_between_15_and_20_annotation_length:>15}")

    # #count how many are above 20% annotation length
    # count_above_20_annotation_length = len(df[df['mpe_annotation_length'] > 20])
    # print(f"{'Above 20% annotation length':<20} {count_above_20_annotation_length:>15}")    

    # #overall mpe annotation length
    # overall_mpe_annotation_length = df['mpe_annotation_length'].mean()
    # print(f"{'Overall mpe annotation length':<20} {overall_mpe_annotation_length:>15}")

    # overall_mae = df['mae'].mean()
    # print(f"{'Overall mae':<20} {overall_mae:>15}")

    # overall_mae_annotation_length = df['mae_annotation_length'].mean()
    # print(f"{'Overall mae annotation length':<20} {overall_mae_annotation_length:>15}")


    # #mean length
    # overall_mean_length = df['Length_fov(mm)'].mean()
    # print(f"{'Overall mean length':<20} {overall_mean_length:>15}")

    # #mean annotation length
    # overall_mean_annotation_length = df['mean_annotation_length'].mean()
    # print(f"{'Overall mean annotation length':<20} {overall_mean_annotation_length:>15}")

    # #expert 1 length
    # overall_expert_1_length = df['Length_1'].mean()
    # print(f"{'Overall expert 1 length':<20} {overall_expert_1_length:>15}")

    # #expert 2 length
    # overall_expert_2_length = df['Length_2'].mean()
    # print(f"{'Overall expert 2 length':<20} {overall_expert_2_length:>15}")

    # #expert 3 length
    # overall_expert_3_length = df['Length_3'].mean()
    # print(f"{'Overall expert 3 length':<20} {overall_expert_3_length:>15}")

   
    #length describe
    # print(df['Length_fov(mm)'].describe())

    #annotation length describe
    # print(df['mean_annotation_length'].describe())  


    # #length 1 describe
    # print(df['Length_1'].describe())

    # #length 2 describe
    # print(df['Length_2'].describe())

    # #length 3 describe
    # print(df['Length_3'].describe())

    # #length 1 describe
    # print(df['Length_1'].describe())



    # print('mpe_annotation')
    # df['mpe_annotation'] =abs(df['Length_fov(mm)']-df['mean_annotation_length'])/df['mean_annotation_length']*10
    # df
    
    # print(df['mpe_annotation'].describe())

#     #mpe annotation median
#     #mpe annotation median
#     print('mpe_annotation median')  
#     print(df['mpe_annotation'].median())
#     #mae annotation
#     df['mae_annotation'] = (df['mean_annotation_length'] - df['Length_fov(mm)'])

#     print(df['mae_annotation'].describe())

#     #print mpe sign 


#     df['mae_with_sign_1'] =  df['Length_1']- df['Length_fov(mm)'] 
#     df['mae_with_sign_2'] = df['Length_2']- df['Length_fov(mm)']
#     df['mae_with_sign_3'] = df['Length_3']- df['Length_fov(mm)'] 

#    #mpe with sign
#     df['mpe_with_sign_1'] = (df['Length_1'] - df['Length_fov(mm)'])/df['Length_1']*100
#     df['mpe_with_sign_2'] = (df['Length_2'] - df['Length_fov(mm)'])/df['Length_2']*100
#     df['mpe_with_sign_3'] = (df['Length_3'] - df['Length_fov(mm)'])/df['Length_3']*100

#     print(df['mpe_with_sign_1'].describe())
#     print(df['mpe_with_sign_2'].describe())
#     print(df['mpe_with_sign_3'].describe())


#     #annotation length with sign

#     df['mean_annotation_length_1'] = df['mean_annotation_length']-df['Length_1']
#     df['mean_annotation_length_2'] = df['mean_annotation_length']-df['Length_2']
#     df['mean_annotation_length_3'] = df['mean_annotation_length']-df['Length_3']

#     print(df['mean_annotation_length_1'].describe())
#     print(df['mean_annotation_length_2'].describe())
#     print(df['mean_annotation_length_3'].describe())
   

#     #mpe with sign annotation length
#     df['mpe_with_sign_annotation_length_1'] = (df['mean_annotation_length'] - df['Length_1'])/df['Length_1']*100
#     df['mpe_with_sign_annotation_length_2'] = (df['mean_annotation_length'] - df['Length_2'])/df['Length_2']*100
#     df['mpe_with_sign_annotation_length_3'] = (df['mean_annotation_length'] - df['Length_3'])/df['Length_3']*100

#     print(df['mpe_with_sign_annotation_length_1'].describe())
#     print(df['mpe_with_sign_annotation_length_2'].describe())
#     print(df['mpe_with_sign_annotation_length_3'].describe())
   
   
   
   
#    #compare scales pred_scale and scale_1
#     df['pred_scale_1'] = df['pred_scale'] - df['Scale_1']
#     df['pred_scale_2'] = df['pred_scale'] - df['Scale_2']
#     df['pred_scale_3'] = df['pred_scale'] - df['Scale_3']

#     print(df['pred_scale_1'].describe())
#     print(df['pred_scale_2'].describe())
#     print(df['pred_scale_3'].describe())




    #mpe with sign scale



    # print ('========mpe with sign scale========')    

    # df['mpe_with_sign_scale_1'] = (df['pred_scale'] - df['Scale_1'])/df['Scale_1']*100
    # df['mpe_with_sign_scale_2'] = (df['pred_scale'] - df['Scale_2'])/df['Scale_2']*100
    # df['mpe_with_sign_scale_3'] = (df['pred_scale'] - df['Scale_3'])/df['Scale_3']*100

    # print(df['mpe_with_sign_scale_1'].describe())
    # print(df['mpe_with_sign_scale_2'].describe())
    # print(df['mpe_with_sign_scale_3'].describe())

    
   
   





    # #need to check how the scale related to height
    # df['pred_scale_height'] = df['pred_scale']*df['Height_fov(mm)']

    





    # #pixel difference 
    # df['pixel_difference_1'] = df['Length_1_pixels'] - df['pred_Distance_pixels']
    # df['pixel_difference_2'] = df['Length_2_pixels'] - df['pred_Distance_pixels']
    # df['pixel_difference_3'] = df['Length_3_pixels'] - df['pred_Distance_pixels']

    # #mpe with sign pixel difference
    # df['mpe_with_sign_pixel_difference_1'] = (df['Length_1_pixels'] - df['pred_Distance_pixels'])/df['Length_1_pixels']*100
    # df['mpe_with_sign_pixel_difference_2'] = (df['Length_2_pixels'] - df['pred_Distance_pixels'])/df['Length_2_pixels']*100
    # df['mpe_with_sign_pixel_difference_3'] = (df['Length_3_pixels'] - df['pred_Distance_pixels'])/df['Length_3_pixels']*100

    # print(df['mpe_with_sign_pixel_difference_1'].describe())
    # print(df['mpe_with_sign_pixel_difference_2'].describe())
    # print(df['mpe_with_sign_pixel_difference_3'].describe())



    # #annotation pixels
    # df['annotation_pixels_1'] = (df['Length_ground_truth_annotation_pixels'] - df['Length_1_pixels'])/df['Length_1_pixels']*100
    # df['annotation_pixels_2'] = (df['Length_ground_truth_annotation_pixels'] - df['Length_2_pixels'])/df['Length_2_pixels']*100
    # df['annotation_pixels_3'] = (df['Length_ground_truth_annotation_pixels'] - df['Length_3_pixels']) /df['Length_3_pixels']*100

    # print(df['annotation_pixels_1'].describe())
    # print(df['annotation_pixels_2'].describe())
    # print(df['annotation_pixels_3'].describe())




    # #find how many bigger than positive 10%
    # count_bigger_than_positive_10 = len(df[df['annotation_pixels_1'] > 10])
    # print(f"{'Bigger than positive 10%':<20} {count_bigger_than_positive_10:>15} out of {len(df):>15}")






    # #find how many bigger than negative 10%
    # count_bigger_than_negative_10 = len(df[df['annotation_pixels_1'] < -10])
    # print(f"{'Bigger than negative 10%':<20} {count_bigger_than_negative_10:>15} out of {len(df):>15}")




    #mae annotaiton pixels with pred_distance
    df['mae_annotation_pixels_1'] = df['pred_Distance_pixels']-df['Length_ground_truth_annotation_pixels'] 
    


    # #create a plotly graph of the annotation pixels difference with points with hover text as label and prawn id

    # import plotly.express as px

    # fig = px.scatter(df, y='mpe_annotation',x='mae_annotation_pixels_1', hover_data=['Label', 'PrawnID'])
    # #save as html
    # fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/analysis/graphs/annotation_pixels_difference_{args.type}_{args.weights_type}.html')


    # the error rate     

    # #print mpe annotation with sign
    # print('mpe annotation with sign')
    # print(df['mpe_annotation'].describe())


    # df['annotation_pixels_1'] = (df['Length_ground_truth_annotation_pixels'] - df['Length_1_pixels'])/df['Length_1_pixels']*100
    # df['annotation_pixels_2'] = (df['Length_ground_truth_annotation_pixels'] - df['Length_2_pixels'])/df['Length_2_pixels']*100
    # df['annotation_pixels_3'] = (df['Length_ground_truth_annotation_pixels'] - df['Length_3_pixels']) /df['Length_3_pixels']*100


    # print(df['annotation_pixels_1'].describe())

    # print(df['annotation_pixels_2'].describe())

    # print(df['annotation_pixels_3'].describe())

    # box_x_min =[]
    # box_y_min =[]
    # COLOR=[]
    # for index, row in df.iterrows():

    #     bbox_3 = ast.literal_eval(row['BoundingBox_3'])
    #     bbox_3 = tuple(float(coord) for coord in bbox_3)

    #     x_min_3 = bbox_3[0]
    #     y_min_3 = bbox_3[1]
    #     width_3 = bbox_3[2]
    #     height_3 = bbox_3[3]

    #     top_left_3 = [x_min_3, y_min_3]
    #     top_right_3 = [x_min_3 + width_3, y_min_3]
    #     bottom_left_3 = [x_min_3, y_min_3 + height_3]
    #     bottom_right_3 = [x_min_3 + width_3, y_min_3 + height_3]   

    #     box_x_min.append(x_min_3)
    #     box_y_min.append(2988-y_min_3)
    #     COLOR.append(row['annotation_pixels_1'])

    #     #graph showing coordinates and color of error rate
    # fig = px.scatter(
    #     df, 
    #     x=box_x_min, 
    #     y=box_y_min, 
    #     color=COLOR,
    #     range_color=[-70,70], 
    #     hover_data=['Label', 'PrawnID'],
    #     width=1200,  # Set a reasonable display width
    #     height=800   # Set a reasonable display height
    # )

    # # Update layout to show full image dimensions
    # fig.update_layout(
    #     xaxis=dict(
    #         range=[0, 5312],  # Set x-axis range to image width
    #         title="Image Width (pixels)"
    #     ),
    #     yaxis=dict(
    #         range=[0, 2988],  # Set y-axis range to image height
    #         title="Image Height (pixels)"
    #     ),
    #     title="Error Distribution Across Image Space"
    # )

    # # Add a colorbar title
    # fig.update_coloraxes(colorbar_title="Error (%)")

    # # Save the plot
    # fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/analysis/graphs/annotation_pixels_difference_{args.type}_{args.weights_type}.html')





    # #annotation pixels
    # df['annotation_pixels_1'] = df['Length_ground_truth_annotation_pixels'] - df['Length_1_pixels']
    # df['annotation_pixels_2'] = df['Length_ground_truth_annotation_pixels'] - df['Length_2_pixels']
    # df['annotation_pixels_3'] = df['Length_ground_truth_annotation_pixels'] - df['Length_3_pixels']

   
    # print(df['annotation_pixels_1'].describe())
    # print(df['annotation_pixels_2'].describe())
    # print(df['annotation_pixels_3'].describe())





    # #mpe with sign pixel difference
    # df['mpe_with_sign_pixel_difference_1'] = (df['Length_1_pixels'] - df['Length_fov(mm)'])/df['Length_1_pixels']*100
    # df['mpe_with_sign_pixel_difference_2'] = (df['Length_2_pixels'] - df['Length_fov(mm)'])/df['Length_2_pixels']*100
    # df['mpe_with_sign_pixel_difference_3'] = (df['Length_3_pixels'] - df['Length_fov(mm)'])/df['Length_3_pixels']*100

    # print(df['mpe_with_sign_pixel_difference_1'].describe())
    # print(df['mpe_with_sign_pixel_difference_2'].describe())
    # print(df['mpe_with_sign_pixel_difference_3'].describe())    

    # #BY POND TYPE
    # for pond_type in df['Pond_Type'].unique():
    #     print(f"\n=== {pond_type} Statistics ===")
    #     print(df[df['Pond_Type'] == pond_type]['mpe_with_sign_pixel_difference_1'].describe())
    #     print(df[df['Pond_Type'] == pond_type]['mpe_with_sign_pixel_difference_2'].describe())
    #     print(df[df['Pond_Type'] == pond_type]['mpe_with_sign_pixel_difference_3'].describe())




   
    # #by pond type
    # for pond_type in df['Pond_Type'].unique():
    #     print(f"\n=== {pond_type} Statistics ===")
    #     print(df[df['Pond_Type'] == pond_type]['Length_fov(mm)'].describe())
    #     print(df[df['Pond_Type'] == pond_type]['mean_annotation_length'].describe())

    #     print(df[df['Pond_Type'] == pond_type]['Length_1'].describe())

    #     print(df[df['Pond_Type'] == pond_type]['Length_2'].describe())

    #     print(df[df['Pond_Type'] == pond_type]['Length_3'].describe())

    #     print(df[df['Pond_Type'] == pond_type]['mae'].describe())

    #     print(df[df['Pond_Type'] == pond_type]['mae_annotation_length'].describe())

    #     print(df[df['Pond_Type'] == pond_type]['mpe'].describe())

    #     print(df[df['Pond_Type'] == pond_type]['mpe_annotation_length'].describe())

    #     print(df[df['Pond_Type'] == pond_type]['mpe_annotation'].describe())

    #     print(df[df['Pond_Type'] == pond_type]['mpe_annotation_length'].describe())


        
        
    #     #mean scale

    #  #overall mpe annotation mpe

    # print('========mpe annotation========')

    # print(df['mpe_annotation'].describe())
    
    # #overall mpe and mae

    # print('========mpe and mae========')

   


    # #overall mpe and mae

    # print('========mpe and mae========')

    # print(df['mpe'].describe())

    # print(df['mae'].describe())



    # for pond_type in df['Pond_Type'].unique():
        
    # #overall pond type
    #     print(f"\n=== {pond_type} Statistics ===")
    
    
    
    #     print(df[df['Pond_Type'] == pond_type]['mpe_annotation'].describe())
       

    #     #overall mae 
    #     print(df[df['Pond_Type'] == pond_type]['mae'].describe())

    #     #overall mpe
    #     print(df[df['Pond_Type'] == pond_type]['mpe'].describe())

    
        
        
        
        
    # #treat pred scale as the scale for length_1 pixels
    # df['Length_1_pixels_pred_scale'] = df['Length_1_pixels']/df['pred_scale']*10
    # df['Length_2_pixels_pred_scale'] = df['Length_2_pixels']/df['pred_scale']*10
    # df['Length_3_pixels_pred_scale'] = df['Length_3_pixels']/df['pred_scale']*10

    # print('========Length_1_pixels_pred_scale========')
    # print(df['Length_1_pixels_pred_scale'].describe())
    # print('========Length_2_pixels_pred_scale========')
    # print(df['Length_2_pixels_pred_scale'].describe())
    # print('========Length_3_pixels_pred_scale========')
    # print(df['Length_3_pixels_pred_scale'].describe())
    
    # #mpe with lenghtfov and pred scale
    # df['mpe_with_lengthfov_and_pred_scale_1'] = abs(df['Length_fov(mm)'] - df['Length_1_pixels_pred_scale'])/df['Length_1_pixels_pred_scale']*100
    # df['mpe_with_lengthfov_and_pred_scale_2'] = abs(df['Length_fov(mm)'] - df['Length_2_pixels_pred_scale'])/df['Length_2_pixels_pred_scale']*100
    # df['mpe_with_lengthfov_and_pred_scale_3'] = abs(df['Length_fov(mm)'] - df['Length_3_pixels_pred_scale'])/df['Length_3_pixels_pred_scale']*100

    # print(df['mpe_with_lengthfov_and_pred_scale_1'].describe())
    # print(df['mpe_with_lengthfov_and_pred_scale_2'].describe())
    # print(df['mpe_with_lengthfov_and_pred_scale_3'].describe())

    # for pond_type in df['Pond_Type'].unique():
    #     print(f"\n=== {pond_type} Statistics ===")
    #     print(df[df['Pond_Type'] == pond_type]['mpe_with_lengthfov_and_pred_scale_1'].describe())
    #     print(df[df['Pond_Type'] == pond_type]['mpe_with_lengthfov_and_pred_scale_2'].describe())
    #     print(df[df['Pond_Type'] == pond_type]['mpe_with_lengthfov_and_pred_scale_3'].describe())




    #count how many are between 10% and 15% annotation length
    return __name__



if __name__ == "__main__":
    main()