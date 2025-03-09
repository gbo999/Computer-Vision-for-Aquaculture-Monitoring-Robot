import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import fiftyone as fo


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

# ----- Error Calculation -----

# Calculate Mean Percentage Error (MPE) for each of the three length measurements
# This compares each measurement against the field of view prediction length
    df['MPE_length1'] = abs(df['Length_1'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['MPE_length2'] = abs(df['Length_2'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['MPE_length3'] = abs(df['Length_3'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['mae_length1'] = abs(df['Length_fov(mm)'] - df['Length_1'])
    df['mae_length2'] = abs(df['Length_fov(mm)'] - df['Length_2'])
    df['mae_length3'] = abs(df['Length_fov(mm)'] - df['Length_3'])
# Determine the minimum MPE across all three measurements for each row
    # This represents the best-case error for each measurement
    if args.error_size == 'min':    
        df['min_mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].min(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].min(axis=1)
    elif args.error_size == 'median':
        df['min_mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].median(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].median(axis=1)
    elif args.error_size == 'mean':
        df['min_mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].mean(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].mean(axis=1)
    elif args.error_size == 'max':
        df['min_mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].max(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].max(axis=1)

# ----- Best Length Determination -----

# Create a mask identifying which length measurement gave the minimum MPE
    min_mpe_mask = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].eq(df['min_mpe'], axis=0)

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
    df['pred_scale'] = df['pred_Distance_pixels'] / df['Length_fov(mm)'] * 10

# Normalize expert measurements to pixels for comparison with predictions
# This converts the best expert measurement to the same pixel scale as predictions
    df['expert_normalized_pixels'] = df.apply(
    lambda row: row['best_length_pixels'] * row['pred_scale'] / row[f'Scale_{min_mpe_index[row.name]}'],
    axis=1
)
    

# Calculate minimum error in pixels (absolute difference between expert and prediction)
    df['min_error_pixels'] = abs(df['expert_normalized_pixels'] - df['pred_Distance_pixels'])

# Calculate minimum MAPE in pixels (percentage error in pixel space)
    df['min_mape_pixels'] = df['min_error_pixels'] / df['pred_Distance_pixels'] * 100

# ----- Flag High Errors -----

# Create a flag for measurements with errors > 10%
    df['high_error'] = df['min_mpe'] > 10

# Calculate and display the percentage of measurements with high errors
    high_error_pct = df['high_error'].mean() * 100
    print(f"Percentage of measurements with errors > 10%: {high_error_pct:.1f}%")

# ----- Additional Pixel Differences -----

# Calculate absolute pixel differences between predicted and expert measurements
    df['pred_pixels_diff'] = abs(df['pred_Distance_pixels'] - df['expert_normalized_pixels'])

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

# ----- Flagging Pixel Errors -----

# Define threshold for high pixel percentage error
    pixel_pct_threshold = 10  # 10% threshold

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



# Calculate average pixel percentage error across all three measurements
    df['avg_pixel_error_pct'] = (df['pixel_diff_pct_1'] + df['pixel_diff_pct_2'] + df['pixel_diff_pct_3']) / 3

# Flag high average pixel error
    df['flag_high_avg_pixel_error'] = df['avg_pixel_error_pct'] > pixel_pct_threshold

# ----- GT-Expert Pixel Differences -----

# Calculate absolute pixel differences between ground truth and each expert measurement
    df['gt_expert_diff_1'] = abs(df['Length_ground_truth_annotation_pixels'] - df['Length_1_pixels'])
    df['gt_expert_diff_2'] = abs(df['Length_ground_truth_annotation_pixels'] - df['Length_2_pixels'])
    df['gt_expert_diff_3'] = abs(df['Length_ground_truth_annotation_pixels'] - df['Length_3_pixels'])

# Calculate percentage differences relative to expert measurements
    df['gt_expert_diff_pct_1'] = df['gt_expert_diff_1'] / df['Length_1_pixels'] * 100
    df['gt_expert_diff_pct_2'] = df['gt_expert_diff_2'] / df['Length_2_pixels'] * 100
    df['gt_expert_diff_pct_3'] = df['gt_expert_diff_3'] / df['Length_3_pixels'] * 100

# ----- Flagging GT-Expert Errors -----

# Define threshold for high GT-Expert pixel percentage error
    gt_expert_pct_threshold = 10  # 10% threshold

# Flag when all three GT-Expert measurements exceed the threshold
    df['flag_all_high_gt_expert_error'] = (
    (df['gt_expert_diff_pct_1'] > gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_2'] > gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_3'] > gt_expert_pct_threshold)
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
    df['flag_high_avg_gt_expert_error'] = df['avg_gt_expert_error_pct'] > gt_expert_pct_threshold

# ----- Potential Error Source Flags -----

# Define flags for potential error sources
# 1. High pixel difference between ground truth and expert (normalized)

# 2. High pixel difference between prediction and expert (percentage)
    df['flag_high_pred_diff'] = df['min_mape_pixels'] > 3

# 3. Low pose evaluation score (if available)
# Check which pose evaluation column exists in the dataset
    if 'pose_eval' in df.columns:
        df['flag_low_pose_eval'] = df['pose_eval'] < 0.85
    elif 'pose_eval_iou' in df.columns:
        df['flag_low_pose_eval'] = df['pose_eval_iou'] < 0.85
    else:
        df['flag_low_pose_eval'] = False
        print("Warning: No pose evaluation column found!")

# 4. High pixel difference between prediction and ground truth annotation
    df['flag_pred_gt_diff'] = df['pred_pixel_gt_diff']/df['Length_ground_truth_annotation_pixels']*100 > 5

# ----- Flag Count and Multiple Error Images -----


# ----- Identify Images with All High Errors -----

# Get a complete list of unique image labels
    all_image_labels = df['Label'].unique()

# Count total measurements per image
    total_measurements_by_image = df.groupby('Label').size()

# Count high error measurements per image
    high_error_df = df[df['min_mpe'] > 10]  # Filter for high errors
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



# # ----- VISUALIZATION SECTION -----

# # ----- 1. Matplotlib/Seaborn Bar Chart -----

#     """
# This visualization creates a conditional bar chart comparing error rates
# when flags are present vs. absent. It helps understand the impact of
# each flag on error rates.
# """
# # Create visualization: Conditional bar chart showing percentage of high errors by flag
#     fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes with specified size

# # For each flag, calculate percentage of high errors
#     flag_cols = ['flag_all_high_pixel_error', 'flag_all_high_gt_expert_error', 'flag_all_high_error_rate_image', 'flag_image_multiple_errors']
#     flag_labels = ['All High Pixel Error', 'All High GT-Expert Error', 'All High Error Rate Image', 'Multiple Errors in Image']
#     flag_true_error_pcts = []  # Will store error percentages when flag is True
#     flag_false_error_pcts = [] # Will store error percentages when flag is False

#     for flag in flag_cols:
#     # Calculate percentage of high errors when flag is True
#         if len(df[df[flag]]) > 0:
#             flag_true_error_pcts.append(df[df[flag]]['high_error'].mean() * 100)
#         else:
#             flag_true_error_pcts.append(0)
    
#     # Calculate percentage of high errors when flag is False
#         if len(df[~df[flag]]) > 0:
#             flag_false_error_pcts.append(df[~df[flag]]['high_error'].mean() * 100)
#         else:
#             flag_false_error_pcts.append(0)

# # Create bar chart with grouped bars
#     x = np.arange(len(flag_labels))  # Positions for the bars
#     width = 0.35  # Width of the bars

# # Create paired bars for each flag - one for when flag is present, one for when absent
#     ax.bar(x - width/2, flag_true_error_pcts, width, label='Flag Present', color='#e74c3c')
#     ax.bar(x + width/2, flag_false_error_pcts, width, label='Flag Absent', color='#2ecc71')

# # Configure the axes and labels
#     ax.set_xticks(x)  # Set tick positions
#     ax.set_xticklabels(flag_labels)  # Set tick labels
#     ax.set_ylabel('Percentage of Measurements with Error >10%')  # Y-axis label
#     ax.set_title('Impact of Error Flags on High Error Rate')  # Plot title
#     ax.legend()  # Add legend for bar colors

# # Add a horizontal line for the overall error rate as reference
#     ax.axhline(y=high_error_pct, color='gray', linestyle='--', 
#            label=f'Overall Error Rate ({high_error_pct:.1f}%)')
#     ax.set_ylim(0, 100)  # Set y-axis range from 0 to 100%

# # Adjust layout and save/display the figure
#     plt.tight_layout()  # Adjust spacing to prevent clipping of labels
#     plt.savefig('error_flags_impact.png', dpi=300)  # Save as high-resolution PNG
#     plt.show()  # Display the plot

# # ----- 2. Correlation Heatmap with Seaborn -----

#     """
# This visualization creates a correlation matrix heatmap to show relationships
# between different flags and MPE. It helps identify which flags are most
# strongly associated with error.
# """
# # Create a DataFrame for correlation analysis
#     corr_matrix = pd.DataFrame()

# # Define columns and labels for the enhanced correlation matrix
#     


# # Add flag columns (binary variables) to correlation matrix
#     for i, flag_col in enumerate(flag_columns):
#         corr_matrix[flag_labels[i]] = df[flag_col].astype(float)

# # Add MPE as a continuous variable
#     corr_matrix['MPE'] = df['min_mpe']

# # Calculate correlation matrix
#     correlation_matrix = corr_matrix.corr()

# # Plot the enhanced correlation heatmap
#     plt.figure(figsize=(10, 8))  # Create figure with specified size
# # Create heatmap with annotations, coolwarm colormap, and value range from -1 to 1
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
#     plt.title('Correlation Matrix: Error Flags and MPE')  # Add title
#     plt.tight_layout()  # Adjust spacing
#     plt.show()  # Display the plot

# # ----- 3. Flag Co-occurrence Heatmap with Seaborn -----

#     """
# This visualization creates a heatmap showing co-occurrence of flags.
# The diagonal shows the percentage of measurements with each flag,
# while off-diagonal cells show the percentage of measurements with one flag
# that also have another flag.
# """
# # Original heatmap showing co-occurrence of flags
#     heatmap_data = pd.DataFrame(index=flag_labels, columns=flag_labels)

# # Populate the heatmap data
#     for i, flag1 in enumerate(flag_columns):
#         for j, flag2 in enumerate(flag_columns):
#             if i == j:
#             # Diagonal: Percentage of measurements with this flag
#                 heatmap_data.iloc[i, j] = df[flag1].mean() * 100  
#             else:
#             # Off-diagonal: Percentage of measurements with flag1 that also have flag2
#                 if df[flag1].sum() > 0:
#                     heatmap_data.iloc[i, j] = df[df[flag1]][flag2].astype(float).mean() * 100
#                 else:
#                     heatmap_data.iloc[i, j] = 0.0

# # Convert to float to ensure compatibility with heatmap
#     heatmap_data = heatmap_data.astype(float)

# # Create the heatmap
#     plt.figure(figsize=(10, 8))  # Create figure with specified size
# # Create heatmap with annotations, YlOrRd colormap, and percentage range from 0 to 100
#     sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f', vmin=0, vmax=100)
#     plt.title('Flag Co-occurrence (%)')  # Add title
#     plt.tight_layout()  # Adjust spacing
#     plt.show()  # Display the plot

    flag_columns = ['flag_all_high_pixel_error', 'flag_all_high_gt_expert_error', 
                        'flag_all_high_error_rate_image', 'flag_image_multiple_errors','high_error']
    


# Create a new comprehensive box plot with mutually exclusive categories
    exclusive_flags_fig = go.Figure()

# Define priorities based on percentage thresholds
# New approach: Add columns with the actual percentage values for each flag

    df['pred_gt_diff_pct'] = df['pred_pixel_gt_diff'] / df['pred_Distance_pixels'] * 100

# Set the pose value to a comparable scale (0-100)
    if 'pose_eval' in df.columns:
        df['pose_pct'] = (1 - df['pose_eval']) * 100  # Convert to percent error (higher = worse)
    elif 'pose_eval_iou' in df.columns:
        df['pose_pct'] = (1 - df['pose_eval_iou']) * 100
    else:
        df['pose_pct'] = 0

# Now determine the primary flag based on the highest percentage
    def get_primary_flag_by_pct(row):
    # Check all other flags first to find the one with highest percentage value
        
        if row['flag_all_high_gt_expert_error'] & row['high_error'] :
            return 'All GT-Expert error >10%'
        elif (row['pose_pct']> 15) & (row['high_error']):
            return 'Pose error >15%'

        elif row['flag_pred_gt_diff'] & row['high_error']:
            return 'Prediction-GT pixel diff >5%'

        elif row['flag_all_high_pixel_error'] & row['high_error']:
            return 'All High Pixel Error'
  
        elif row['flag_pred_gt_diff'] & row['high_error']:
            return 'Prediction-GT pixel diff >5%'

        elif row['flag_all_high_error_rate_image'] :
          print("All High error rate image")
          return 'All High error rate image'
    


        elif row['flag_image_multiple_errors'] & row['high_error']:
            return 'Multiple Errors in Same Image'
    
    # If no flags at all
        else:
            return 'No Flags'

# Assign each measurement to exactly one category based on highest percentage
    df['assigned_category'] = df.apply(get_primary_flag_by_pct, axis=1)

# Create a category for measurements with no flags
    df.loc[df['assigned_category'].isna(), 'assigned_category'] = 'No Flags'

# The order for visualization still needs to be defined
    priority_names = [
    'Pose error >15%',
    'All GT-Expert error >10%',
    'All High Pixel Error',
    'Prediction-GT pixel diff >3%',
    'All High error rate image',
    'Multiple Errors in Same Image',
]

# Print some statistics about our prioritized assignments
    print("\nAssignments based on percentage values:")
    for category in priority_names + ['No Flags']:
        count = len(df[df['assigned_category'] == category])
        print(f"{category}: {count} measurements")

# Store colors for visualization
    priority_colors = ['#9b59b6', '#2ecc71', '#f39c12', '#3498db', '#e74c3c']
    df['flag_count'] = 0
    for flag in flag_columns:
        df['flag_count'] += df[flag].astype(int) 

    df['min_gt_diff']=0
    df['gt_diff_pct']=0
# Add a box plot for each exclusive category
    categories = [cat for cat in priority_names] + ['No Flags']
    for i, category in enumerate(categories):
        cat_df = df[df['assigned_category'] == category]
    
        if len(cat_df) > 0:  # Only add if there are measurements in this category
            color = priority_colors[i] if i < len(priority_colors) else 'gray'
            
            # Create separate traces for each pond type
            for pond_type in cat_df['Pond_Type'].unique():
                pond_df = cat_df[cat_df['Pond_Type'] == pond_type]
                
                # Define marker symbol based on pond type
                symbol = ('circle' if pond_type == 'square' else 
                         'square' if pond_type == 'circle_female' else 
                         'diamond' if pond_type == 'circle_male' else 
                         'cross')
                
                exclusive_flags_fig.add_trace(go.Box(
                    y=pond_df['min_mpe'],
                    name=category,
                    boxmean=True,
                    marker_color=color,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.5,
                    width=0.6,
                    quartilemethod="linear",
                    marker=dict(
                        symbol=symbol,
                        size=8
                    ),
                    showlegend=True if pond_type == cat_df['Pond_Type'].iloc[0] else False,  # Show legend only once per category
                    legendgroup=category,  # Group traces by category
                    hovertemplate=
                        f"<b>{category}</b><br>" +
                        "<b>ID:</b> %{customdata[0]}<br>" +
                        "<b>Image:</b> %{customdata[1]}<br>" +
                        "<b>Pond Type:</b> %{customdata[9]}<br>" +
                        "<b>Error:</b> %{y:.1f}%<br>" +
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
                        "<b>pose error</b> %{customdata[18]:.1f}%<br>",
                    customdata=pond_df[['PrawnID', 'Label', 'min_gt_diff', 'flag_count', 
                                    'gt_diff_pct', 'pred_pixel_gt_diff', 'pred_gt_diff_pct',
                                    'min_error_pixels', 'min_mape_pixels','Pond_Type','best_length_pixels',
                                    'pred_Distance_pixels','pixel_diff_pct_1','pixel_diff_pct_2',
                                    'pixel_diff_pct_3','gt_expert_diff_pct_1','gt_expert_diff_pct_2',
                                    'gt_expert_diff_pct_3','pose_pct']].values
                ))

# Add horizontal line at 10% error
    exclusive_flags_fig.add_shape(
    type='line',
    x0=-0.5, x1=len(categories) - 0.5,
    y0=10, y1=10,
    line=dict(color='red', dash='dash')
)

# Update layout
    exclusive_flags_fig.update_layout(
    title='Error Distribution by Exclusive Flag Categories (Prioritizing Multiple Errors in Image)',
    yaxis_title='Min MPE (%)',
    height=600, width=1000,
    boxmode='group',
    yaxis=dict(
        range=[0, max(50, df['min_mpe'].max() * 1.1)]
    ),
    margin=dict(l=50, r=50, t=80, b=120)
)

# Add counts to the box plot names
    for i, trace in enumerate(exclusive_flags_fig.data):
        category = trace.name
        count = len(df[df['assigned_category'] == category])
        trace.name = f"{category} (n={count})"

    exclusive_flags_fig.show()

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

    mae_flags = df[df['flag_count'] == 0]['mae'].mean()
    mape_flags = df[df['flag_count'] == 0]['min_mpe'].mean()
    mae_by_pond_type = df[df['flag_count'] == 0].groupby('Pond_Type')['mae'].mean()
    mape_by_pond_type = df[df['flag_count'] == 0].groupby('Pond_Type')['min_mpe'].mean()
    
    mae_with_flags = df[df['flag_count'] > 0]['mae'].mean()
    mape_with_flags = df[df['flag_count'] > 0]['min_mpe'].mean()
    mae_by_pond_type_with_flags = df[df['flag_count'] > 0].groupby('Pond_Type')['mae'].mean()
    mape_by_pond_type_with_flags = df[df['flag_count'] > 0].groupby('Pond_Type')['min_mpe'].mean()

    # Print overall statistics table
    print("\n=== Overall Statistics ===")
    print(f"{'Metric':<30} {'Without Flags':>15} {'With Flags':>15}")
    print("-" * 60)
    print(f"{'MAE (Mean Absolute Error)':<30} {mae_flags:>15.2f} {mae_with_flags:>15.2f}")
    print(f"{'MAPE (Mean Absolute % Error)':<30} {mape_flags:>15.2f} {mape_with_flags:>15.2f}")

    # Print MAE by pond type table
    print("\n=== MAE by Pond Type ===")
    print(f"{'Pond Type':<20} {'Without Flags':>15} {'With Flags':>15}")
    print("-" * 50)
    for pond_type in mae_by_pond_type.index:
        print(f"{pond_type:<20} {mae_by_pond_type[pond_type]:>15.2f} {mae_by_pond_type_with_flags[pond_type]:>15.2f}")

    # Print MAPE by pond type table
    print("\n=== MAPE by Pond Type ===")
    print(f"{'Pond Type':<20} {'Without Flags':>15} {'With Flags':>15}")
    print("-" * 50)
    for pond_type in mape_by_pond_type.index:
        print(f"{pond_type:<20} {mape_by_pond_type[pond_type]:>15.2f} {mape_by_pond_type_with_flags[pond_type]:>15.2f}")

    # Print sample counts
    print("\n=== Sample Counts ===")
    print(f"{'Pond Type':<20} {'Without Flags':>15} {'With Flags':>15} {'Total':>15}")
    print("-" * 65)
    for pond_type in df['Pond_Type'].unique():
        without_flags = len(df[(df['Pond_Type'] == pond_type) & (df['flag_count'] == 0)])
        with_flags = len(df[(df['Pond_Type'] == pond_type) & (df['flag_count'] > 0)])
        total = without_flags + with_flags
        print(f"{pond_type:<20} {without_flags:>15} {with_flags:>15} {total:>15}")

    total_without_flags = len(df[df['flag_count'] == 0])
    total_with_flags = len(df[df['flag_count'] > 0])
    print("-" * 65)
    print(f"{'Total':<20} {total_without_flags:>15} {total_with_flags:>15} {len(df):>15}")


    #count statistics for each flag and pond type using primary flag using the assigned category
    for category in categories:
        print(f"\n=== {category} Statistics ===")
        print(f"{'Pond Type':<20} {'Count':>15}")
        print("-" * 30)
        
        for pond_type in df['Pond_Type'].unique():
            count = len(df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] == category)])
            print(f"{pond_type:<20} {count:>15}")

        
        
        
        
    
    






    return __name__



if __name__ == "__main__":
    main()