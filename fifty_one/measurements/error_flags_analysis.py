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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# ----- Data Loading and Preprocessing -----

# Load the dataset from Excel file
df = pd.read_excel(r'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/updated_filtered_data_with_lengths_body-all.xlsx')

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
# This compares each measurement against the field of view length (ground truth)
df['MPE_length1'] = abs(df['Length_1'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
df['MPE_length2'] = abs(df['Length_2'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
df['MPE_length3'] = abs(df['Length_3'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100

# Determine the minimum MPE across all three measurements for each row
# This represents the best-case error for each measurement
df['min_mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].min(axis=1)

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
df['flag_high_gt_diff'] = df['min_gt_diff']/df['expert_normalized_pixels']*100 > 3

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

# Count the number of flags for each measurement
# This helps identify measurements with multiple potential error sources
df['flag_count'] = df[['flag_high_gt_diff', 'flag_high_pred_diff', 
                       'flag_low_pose_eval', 'flag_pred_gt_diff']].sum(axis=1)

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

# ----- Create text descriptions for flags -----

def get_flag_descriptions(row):
    """
    Generate a formatted text description of all flags that are True for a given row.
    
    Parameters:
    -----------
    row : pandas.Series
        A row from the DataFrame containing flag columns
        
    Returns:
    --------
    str
        Formatted text listing all active flags
    """
    flags = []
    
    # Check each flag and add its description if True
    if row['flag_high_gt_diff']:
        flags.append("High GT-Expert Diff")
    
    if row['flag_high_pred_diff']:
        flags.append("High Pred-Expert Diff")
    
    if row['flag_low_pose_eval']:
        flags.append("Low Pose Eval")
    
    if row['flag_pred_gt_diff']:
        flags.append("High Pred-GT Diff")
    
    if row['flag_image_multiple_errors']:
        flags.append("Multiple Errors in Image")
    
    # Return formatted text or "None" if no flags
    return "<br>".join(flags) if flags else "None"

# Add a column with human-readable flag information for use in visualizations
df['flag_info'] = df.apply(get_flag_descriptions, axis=1)

# ----- VISUALIZATION SECTION -----

# ----- 1. Matplotlib/Seaborn Bar Chart -----

"""
This visualization creates a conditional bar chart comparing error rates
when flags are present vs. absent. It helps understand the impact of
each flag on error rates.
"""
# Create visualization: Conditional bar chart showing percentage of high errors by flag
fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes with specified size

# For each flag, calculate percentage of high errors
flag_cols = ['flag_high_gt_diff', 'flag_high_pred_diff', 'flag_low_pose_eval', 'flag_pred_gt_diff']
flag_labels = ['High GT Diff\n(>20px)', 'High Pred Diff\n(>20px)', 'Low Pose Eval\n(<0.85)', 'High Pred GT Diff\n(>20px)']

flag_true_error_pcts = []  # Will store error percentages when flag is True
flag_false_error_pcts = [] # Will store error percentages when flag is False

for flag in flag_cols:
    # Calculate percentage of high errors when flag is True
    if len(df[df[flag]]) > 0:
        flag_true_error_pcts.append(df[df[flag]]['high_error'].mean() * 100)
    else:
        flag_true_error_pcts.append(0)
    
    # Calculate percentage of high errors when flag is False
    if len(df[~df[flag]]) > 0:
        flag_false_error_pcts.append(df[~df[flag]]['high_error'].mean() * 100)
    else:
        flag_false_error_pcts.append(0)

# Create bar chart with grouped bars
x = np.arange(len(flag_labels))  # Positions for the bars
width = 0.35  # Width of the bars

# Create paired bars for each flag - one for when flag is present, one for when absent
ax.bar(x - width/2, flag_true_error_pcts, width, label='Flag Present', color='#e74c3c')
ax.bar(x + width/2, flag_false_error_pcts, width, label='Flag Absent', color='#2ecc71')

# Configure the axes and labels
ax.set_xticks(x)  # Set tick positions
ax.set_xticklabels(flag_labels)  # Set tick labels
ax.set_ylabel('Percentage of Measurements with Error >10%')  # Y-axis label
ax.set_title('Impact of Error Flags on High Error Rate')  # Plot title
ax.legend()  # Add legend for bar colors

# Add a horizontal line for the overall error rate as reference
ax.axhline(y=high_error_pct, color='gray', linestyle='--', 
           label=f'Overall Error Rate ({high_error_pct:.1f}%)')
ax.set_ylim(0, 100)  # Set y-axis range from 0 to 100%

# Adjust layout and save/display the figure
plt.tight_layout()  # Adjust spacing to prevent clipping of labels
plt.savefig('error_flags_impact.png', dpi=300)  # Save as high-resolution PNG
plt.show()  # Display the plot

# ----- 2. Correlation Heatmap with Seaborn -----

"""
This visualization creates a correlation matrix heatmap to show relationships
between different flags and MPE. It helps identify which flags are most
strongly associated with error.
"""
# Create a DataFrame for correlation analysis
corr_matrix = pd.DataFrame()

# Define columns and labels for the enhanced correlation matrix
flag_columns = ['flag_high_gt_diff', 'flag_high_pred_diff', 
                'flag_low_pose_eval', 'flag_pred_gt_diff']
flag_labels = ['High GT Diff', 'High Pred Diff', 
               'Low Pose Eval', 'High Pred-GT Diff']

# Add additional variables for enhanced analysis
enhanced_flag_columns = flag_columns + ['flag_image_multiple_errors']
enhanced_flag_labels = flag_labels + ['Multiple Errors in Image']

# Add flag columns (binary variables) to correlation matrix
for i, flag_col in enumerate(flag_columns + ['flag_image_multiple_errors']):
    corr_matrix[enhanced_flag_labels[i]] = df[flag_col].astype(float)

# Add MPE as a continuous variable
corr_matrix['MPE'] = df['min_mpe']

# Calculate correlation matrix
correlation_matrix = corr_matrix.corr()

# Plot the enhanced correlation heatmap
plt.figure(figsize=(10, 8))  # Create figure with specified size
# Create heatmap with annotations, coolwarm colormap, and value range from -1 to 1
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix: Error Flags and MPE')  # Add title
plt.tight_layout()  # Adjust spacing
plt.show()  # Display the plot

# ----- 3. Flag Co-occurrence Heatmap with Seaborn -----

"""
This visualization creates a heatmap showing co-occurrence of flags.
The diagonal shows the percentage of measurements with each flag,
while off-diagonal cells show the percentage of measurements with one flag
that also have another flag.
"""
# Original heatmap showing co-occurrence of flags
heatmap_data = pd.DataFrame(index=flag_labels, columns=flag_labels)

# Populate the heatmap data
for i, flag1 in enumerate(flag_columns):
    for j, flag2 in enumerate(flag_columns):
        if i == j:
            # Diagonal: Percentage of measurements with this flag
            heatmap_data.iloc[i, j] = df[flag1].mean() * 100  
        else:
            # Off-diagonal: Percentage of measurements with flag1 that also have flag2
            if df[flag1].sum() > 0:
                heatmap_data.iloc[i, j] = df[df[flag1]][flag2].astype(float).mean() * 100
            else:
                heatmap_data.iloc[i, j] = 0.0

# Convert to float to ensure compatibility with heatmap
heatmap_data = heatmap_data.astype(float)

# Create the heatmap
plt.figure(figsize=(10, 8))  # Create figure with specified size
# Create heatmap with annotations, YlOrRd colormap, and percentage range from 0 to 100
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f', vmin=0, vmax=100)
plt.title('Flag Co-occurrence (%)')  # Add title
plt.tight_layout()  # Adjust spacing
plt.show()  # Display the plot

# ----- 4. Plotly Scatter Plot -----

"""
This interactive visualization creates a scatter plot showing the relationship
between ground truth pixel difference and MPE, with points colored by flag count
and shaped by pond type. It includes rich hover information.
"""
# Define marker shapes for different pond types
pond_shapes = {
    'circle_male': 'circle', 
    'circle_female': 'diamond', 
    'square': 'square'
}

# Create figure using Plotly Graph Objects for more control
fig = go.Figure()

# Add traces for each pond type with improved hover information
for pond_type, shape in pond_shapes.items():
    # Filter data for the current pond type
    pond_df = df[df['Pond_Type'] == pond_type]
    
    # Add a scatter trace for this pond type
    fig.add_trace(go.Scatter(
        x=pond_df['min_gt_diff'],  # X-axis: Ground truth difference
        y=pond_df['min_mpe'],      # Y-axis: Mean percentage error
        mode='markers',            # Display as scatter points
        marker=dict(
            size=10,               # Point size
            symbol=shape,          # Shape based on pond type
            color=pond_df['flag_count'],  # Color by number of flags
            colorscale='Viridis',  # Color scale
            # Only show color bar for the first pond type to avoid duplication
            colorbar=dict(title='Number of Flags') if pond_type == list(pond_shapes.keys())[0] else None,
            showscale=pond_type == list(pond_shapes.keys())[0]
        ),
        name=pond_type,            # Name in legend
        text=pond_df['flag_info'], # Text for hover info
        # Custom hover template with HTML formatting
        hovertemplate=
            "<b>ID:</b> %{customdata[0]}<br>" +
            "<b>Image:</b> %{customdata[1]}<br>" +
            "<b>Pond Type:</b> " + pond_type + "<br>" +
            "<b>Error (%):</b> %{customdata[2]:.1f}%<br>" +
            "<b>GT Diff (px):</b> %{customdata[3]:.1f}px<br>" +
            "<b>Flags:</b><br>%{text}<br>" +
            "<extra></extra>",  # Hide secondary box
        # Custom data for hover template
        customdata=pond_df[['PrawnID', 'Label', 'Pond_Type', 'min_mpe', 'min_gt_diff']].values
    ))

# Add horizontal line at 10% error threshold
fig.add_shape(
    type='line',
    x0=0, x1=df['min_gt_diff'].max() * 1.1,  # Line spans full x-axis range
    y0=10, y1=10,                           # Flat line at y=10
    line=dict(color='red', dash='dash')     # Red dashed line
)

# Update layout with titles and labels
fig.update_layout(
    title='Error vs Ground Truth Difference, Colored by Number of Flags',
    xaxis_title='Ground Truth Pixel Difference',
    yaxis_title='Min MPE (%)',
    legend_title='Pond Type',
    height=600, width=900  # Set dimensions
)

fig.show()  # Display the interactive plot

# ----- 5. Plotly Box Plot -----

"""
This visualization creates box plots to compare error distributions between
measurements from images with multiple errors and those without. It includes
individual data points and hover information.
"""
# Create a more detailed box plot to show all errors by flag
box_fig = go.Figure()

# Add box plot for measurements from images with multiple errors
box_fig.add_trace(go.Box(
    y=df[df['flag_image_multiple_errors']]['min_mpe'],  # Y values for first group
    name='From Images with Multiple Errors',            # Name in legend
    boxmean=True,                     # Show mean line
    marker_color='#e74c3c',           # Red color for box
    boxpoints='all',                  # Show all individual points
    jitter=0.3,                       # Add horizontal jitter to points
    pointpos=-1.5,                    # Position points to the left of box
    # Custom hover template with HTML formatting
    hovertemplate=
        "<b>ID:</b> %{customdata[0]}<br>" +
        "<b>Image:</b> %{customdata[1]}<br>" +
        "<b>Error:</b> %{y:.1f}%<br>" +
        "<b>GT Diff:</b> %{customdata[2]:.1f}px<br>" +
        "<extra></extra>",            # Hide secondary box
    # Custom data for hover template
    customdata=df[df['flag_image_multiple_errors']][['PrawnID', 'Label', 'min_gt_diff']].values
))

# Add box plot for measurements from images without multiple errors
box_fig.add_trace(go.Box(
    y=df[~df['flag_image_multiple_errors']]['min_mpe'],  # Y values for second group
    name='From Images without Multiple Errors',          # Name in legend
    boxmean=True,                     # Show mean line
    marker_color='#2ecc71',           # Green color for box
    boxpoints='all',                  # Show all individual points
    jitter=0.3,                       # Add horizontal jitter to points
    pointpos=-1.5,                    # Position points to the left of box
    # Custom hover template
    hovertemplate=
        "<b>ID:</b> %{customdata[0]}<br>" +
        "<b>Image:</b> %{customdata[1]}<br>" +
        "<b>Error:</b> %{y:.1f}%<br>" +
        "<b>GT Diff:</b> %{customdata[2]:.1f}px<br>" +
        "<extra></extra>",
    # Custom data for hover template
    customdata=df[~df['flag_image_multiple_errors']][['PrawnID', 'Label', 'min_gt_diff']].values
))

# Add horizontal line at 10% error threshold
box_fig.add_shape(
    type='line',
    x0=-0.5, x1=1.5,                 # Line spans full width of plot
    y0=10, y1=10,                    # Flat line at y=10
    line=dict(color='red', dash='dash')  # Red dashed line
)

# Update layout with titles and labels
box_fig.update_layout(
    title='Error Distribution by Multiple Errors in Image Flag',
    yaxis_title='Min MPE (%)',
    showlegend=True,
    height=600, width=700  # Set dimensions
)

box_fig.show()  # Display the interactive plot

# ----- 6. Plotly Bar Chart with Subplots -----

"""
This visualization creates a side-by-side comparison of flag count distributions
for measurements with low vs. high errors using bar charts in subplots.
"""
# Create a subplot with 1 row and 2 columns
fig = make_subplots(
    rows=1, cols=2,                  # 1x2 grid of subplots 
    subplot_titles=(                 # Titles for each subplot
        "Measurements with Error ≤10%", 
        "Measurements with Error >10%"
    ),
    column_widths=[0.5, 0.5]         # Equal width for both columns
)

# Count number of measurements with each flag count for low errors
low_error_counts = df[~df['high_error']]['flag_count'].value_counts().sort_index()
low_error_labels = [f"{count} flags" for count in low_error_counts.index]

# Count number of measurements with each flag count for high errors
high_error_counts = df[df['high_error']]['flag_count'].value_counts().sort_index()
high_error_labels = [f"{count} flags" for count in high_error_counts.index]

# Add bar chart for low errors to first subplot
fig.add_trace(
    go.Bar(
        x=low_error_labels,           # X-axis: flag count labels
        y=low_error_counts.values,    # Y-axis: number of measurements
        name="Error ≤10%",            # Name in legend
        marker_color='#2ecc71'        # Green color for bars
    ),
    row=1, col=1                      # Place in first column
)

# Add bar chart for high errors to second subplot
fig.add_trace(
    go.Bar(
        x=high_error_labels,          # X-axis: flag count labels 
        y=high_error_counts.values,   # Y-axis: number of measurements
        name="Error >10%",            # Name in legend
        marker_color='#e74c3c'        # Red color for bars
    ),
    row=1, col=2                      # Place in second column
)

# Update layout with titles and dimensions
fig.update_layout(
    height=500, width=900,           # Set dimensions
    title_text="Distribution of Flag Counts by Error Level",
    showlegend=False                 # Hide legend (redundant with subplot titles)
)

# Set Y-axis titles for both subplots
fig.update_yaxes(title_text="Number of Measurements", row=1, col=1)
fig.update_yaxes(title_text="Number of Measurements", row=1, col=2)

fig.show()  # Display the interactive plot

