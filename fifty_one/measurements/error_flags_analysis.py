import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
def calculate_mape(estimated_lengths, true_lengths):
    """
    Calculate Mean Absolute Percentage Error
    
    Parameters:
    -----------
    estimated_lengths : array-like
        Estimated length measurements (Len_e)
    true_lengths : array-like
        True length measurements (Len_t)
        
    Returns:
    --------
    float
        Mean Absolute Percentage Error (%)
    """
    # Calculate individual absolute percentage errors
    absolute_percentage_errors = [abs(est - true) / est * 100 for est, true in zip(estimated_lengths, true_lengths)]
    
    # Calculate mean of absolute percentage errors
    
    return absolute_percentage_errors

# Load the data
df = pd.read_excel(r'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/updated_filtered_data_with_lengths_body-all.xlsx')

# Clean data
df = df.dropna()

# Replace pond types for consistency
df['Pond_Type'] = df['Pond_Type'].replace({
    'car': 'square',
    'right': 'circle_female', 
    'left': 'circle_male',
})

# First calculate MPE for each length measurement
df['MPE_length1'] = abs(df['Length_1'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
df['MPE_length2'] = abs(df['Length_2'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
df['MPE_length3'] = abs(df['Length_3'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100

# Then find which length gave the minimum MPE
df['min_mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].min(axis=1)

# Create a mask that identifies which length measurement gave the minimum MPE
min_mpe_mask = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].eq(df['min_mpe'], axis=0)

# Instead of extracting numbers, let's map the column names to their corresponding indices
column_to_index = {
    'MPE_length1': 1,
    'MPE_length2': 2,
    'MPE_length3': 3
}


df['pred_scale']=df['pred_Distance_pixels']/df['Length_fov(mm)']*10

# Get the column name that has the minimum MPE for each row
min_mpe_column = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].idxmin(axis=1)

# Map to the correct index (1, 2, or 3)
min_mpe_index = min_mpe_column.map(column_to_index)

# Now we can use this index to get the corresponding length
df['best_length'] = df.apply(lambda row: row[f'Length_{min_mpe_index[row.name]}'], axis=1)


df['best_length_pixels'] = df.apply(lambda row: row[f'Length_{min_mpe_index[row.name]}_pixels'], axis=1)




df['expert_normalized_pixels']=df.apply(lambda row: row['best_length_pixels']*row['pred_scale']/row[f'Scale_{min_mpe_index[row.name]}'],axis=1 )


# #differente between pred pixels and expert measurement pixels
# df['pred_pixels_diff_1']=abs( df['pred_Distance_pixels'] - df['Length_1_pixels'])
# df['pred_pixels_diff_2']=abs(df['pred_Distance_pixels'] - df['Length_2_pixels'])
# df['pred_pixels_diff_3']=abs(df['pred_Distance_pixels'] - df['Length_3_pixels'])


#min error in pixels
df['min_error_pixels']=abs(df['expert_normalized_pixels']-df['pred_Distance_pixels'])



# #mape in pixels 
# df['mape_pixels_1']=calculate_mape(df['pred_Distance_pixels'], df['Length_1_pixels'])
# df['mape_pixels_2']=calculate_mape(df['pred_Distance_pixels'], df['Length_2_pixels'])
# df['mape_pixels_3']=calculate_mape(df['pred_Distance_pixels'], df['Length_3_pixels'])


#min mape in pixels
df['min_mape_pixels']=df['min_error_pixels']/df['pred_Distance_pixels']*100





# # Calculate pixel differences
# df['min_expert_pixels'] = df[['Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].min(axis=1)
# df['max_expert_pixels'] = df[['Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].max(axis=1)
# df['expert_range_pixels'] = df['max_expert_pixels'] - df['min_expert_pixels']

df['pred_pixels_diff'] = abs(df['pred_Distance_pixels'] - df['expert_normalized_pixels'])

df['pred_pixel_gt_diff'] = abs(df['Length_ground_truth_annotation_pixels'] - df['pred_Distance_pixels'])

# Flag measurements with errors > 10%
df['high_error'] = df['min_mpe'] > 10

# Calculate proportion of high errors
high_error_pct = df['high_error'].mean() * 100
print(f"Percentage of measurements with errors > 10%: {high_error_pct:.1f}%")



df['min_gt_diff'] = abs(df['expert_normalized_pixels'] - df['Length_ground_truth_annotation_pixels'])


# Calculate pixel differences for each measurement
df['pixel_diff_1'] = abs(df['pred_Distance_pixels'] - df['Length_1_pixels']) 
df['pixel_diff_2'] = abs(df['pred_Distance_pixels'] - df['Length_2_pixels'])
df['pixel_diff_3'] = abs(df['pred_Distance_pixels'] - df['Length_3_pixels'])

# Calculate pixel percentage differences for each measurement
df['pixel_diff_pct_1'] = df['pixel_diff_1'] / df['Length_1_pixels'] * 100
df['pixel_diff_pct_2'] = df['pixel_diff_2'] / df['Length_2_pixels'] * 100
df['pixel_diff_pct_3'] = df['pixel_diff_3'] / df['Length_3_pixels'] * 100

# Define a threshold for high pixel percentage error
pixel_pct_threshold = 10  # Adjust as needed (e.g., 10%)

# Create a flag for when all three measurements exceed the threshold
df['flag_all_high_pixel_error'] = (
    (df['pixel_diff_pct_1'] > pixel_pct_threshold) & 
    (df['pixel_diff_pct_2'] > pixel_pct_threshold) & 
    (df['pixel_diff_pct_3'] > pixel_pct_threshold)
)

# Create a flag for when any of the three measurements exceed the threshold
df['flag_any_high_pixel_error'] = (
    (df['pixel_diff_pct_1'] > pixel_pct_threshold) | 
    (df['pixel_diff_pct_2'] > pixel_pct_threshold) | 
    (df['pixel_diff_pct_3'] > pixel_pct_threshold)
)

# Calculate average pixel percentage error
df['avg_pixel_error_pct'] = (df['pixel_diff_pct_1'] + df['pixel_diff_pct_2'] + df['pixel_diff_pct_3']) / 3

# Create flag for high average pixel error
df['flag_high_avg_pixel_error'] = df['avg_pixel_error_pct'] > pixel_pct_threshold

# Print statistics about the new flags
print(f"Measurements with all pixel errors > {pixel_pct_threshold}%: {df['flag_all_high_pixel_error'].sum()} ({df['flag_all_high_pixel_error'].mean()*100:.1f}%)")
print(f"Measurements with any pixel error > {pixel_pct_threshold}%: {df['flag_any_high_pixel_error'].sum()} ({df['flag_any_high_pixel_error'].mean()*100:.1f}%)")
print(f"Measurements with avg pixel error > {pixel_pct_threshold}%: {df['flag_high_avg_pixel_error'].sum()} ({df['flag_high_avg_pixel_error'].mean()*100:.1f}%)") 

# Calculate GT-Expert pixel differences for each measurement
df['gt_expert_diff_1'] = abs(df['Length_ground_truth_annotation_pixels'] - df['Length_1_pixels'])
df['gt_expert_diff_2'] = abs(df['Length_ground_truth_annotation_pixels'] - df['Length_2_pixels'])
df['gt_expert_diff_3'] = abs(df['Length_ground_truth_annotation_pixels'] - df['Length_3_pixels'])

# Calculate GT-Expert pixel percentage differences for each measurement
df['gt_expert_diff_pct_1'] = df['gt_expert_diff_1'] / df['Length_1_pixels'] * 100
df['gt_expert_diff_pct_2'] = df['gt_expert_diff_2'] / df['Length_2_pixels'] * 100
df['gt_expert_diff_pct_3'] = df['gt_expert_diff_3'] / df['Length_3_pixels'] * 100

# Define a threshold for high GT-Expert pixel percentage error
gt_expert_pct_threshold = 10  # Adjust as needed (e.g., 10%)

# Create a flag for when all three GT-Expert measurements exceed the threshold
df['flag_all_high_gt_expert_error'] = (
    (df['gt_expert_diff_pct_1'] > gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_2'] > gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_3'] > gt_expert_pct_threshold)
)

# Create a flag for when any of the three GT-Expert measurements exceed the threshold
df['flag_any_high_gt_expert_error'] = (
    (df['gt_expert_diff_pct_1'] > gt_expert_pct_threshold) | 
    (df['gt_expert_diff_pct_2'] > gt_expert_pct_threshold) | 
    (df['gt_expert_diff_pct_3'] > gt_expert_pct_threshold)
)

# Calculate average GT-Expert pixel percentage error
df['avg_gt_expert_error_pct'] = (df['gt_expert_diff_pct_1'] + df['gt_expert_diff_pct_2'] + df['gt_expert_diff_pct_3']) / 3

# Create flag for high average GT-Expert pixel error
df['flag_high_avg_gt_expert_error'] = df['avg_gt_expert_error_pct'] > gt_expert_pct_threshold

# Print statistics about the new GT-Expert flags
print(f"Measurements with all GT-Expert errors > {gt_expert_pct_threshold}%: {df['flag_all_high_gt_expert_error'].sum()} ({df['flag_all_high_gt_expert_error'].mean()*100:.1f}%)")
print(f"Measurements with any GT-Expert error > {gt_expert_pct_threshold}%: {df['flag_any_high_gt_expert_error'].sum()} ({df['flag_any_high_gt_expert_error'].mean()*100:.1f}%)")
print(f"Measurements with avg GT-Expert error > {gt_expert_pct_threshold}%: {df['flag_high_avg_gt_expert_error'].sum()} ({df['flag_high_avg_gt_expert_error'].mean()*100:.1f}%)")




# Define flags for potential error sources
# 1. High pixel difference between ground truth and expert
df['flag_high_gt_diff'] = df['min_gt_diff']/df['expert_normalized_pixels']*100 > 3

# # 2. High variability between expert measurements
# df['flag_high_expert_var'] = df['expert_range_pixels'] > 15

# 3. High pixel difference between prediction and expert 
df['flag_high_pred_diff'] = df['min_mape_pixels']  > 3

# 4. Low pose evaluation score (if available)
if 'pose_eval' in df.columns:
    df['flag_low_pose_eval'] = df['pose_eval'] < 0.85
elif 'pose_eval_iou' in df.columns:
    df['flag_low_pose_eval'] = df['pose_eval_iou'] < 0.85
else:
    df['flag_low_pose_eval'] = False
    print("Warning: No pose evaluation column found!")

df['flag_pred_gt_diff'] = df['pred_pixel_gt_diff']/df['expert_normalized_pixels']*100 > 3

# Count how many flags each measurement has
df['flag_count'] = df[['flag_high_gt_diff',  'flag_high_pred_diff', 'flag_low_pose_eval', 'flag_pred_gt_diff']].sum(axis=1)

# Calculate images with multiple high errors (this is where we make the key change)
# First count how many high errors per image
# Get the complete list of image labels
all_image_labels = df['Label'].unique()

# Calculate total measurements per image
total_measurements_by_image = df.groupby('Label').size()

# Calculate high error counts per image
high_error_df = df[df['min_mpe'] > 10]  # Filter for high errors
high_error_counts_by_image = high_error_df.groupby('Label').size()

# Reindex both Series to ensure they have the same index
total_measurements_by_image = total_measurements_by_image.reindex(all_image_labels, fill_value=0)
high_error_counts_by_image = high_error_counts_by_image.reindex(all_image_labels, fill_value=0)

# Now they have the same index and can be compared safely
image_100_error_rate = high_error_counts_by_image[
    (high_error_counts_by_image == total_measurements_by_image) & 
    (total_measurements_by_image > 1)
].index.tolist()

# Flag these images in the DataFrame
df['flag_all_high_error_rate_image'] = df['Label'].isin(image_100_error_rate)

images_with_multiple_high_errors = high_error_counts_by_image[high_error_counts_by_image > 1].index.tolist()

df['flag_image_multiple_errors'] = df['Label'].isin(images_with_multiple_high_errors)


# Add to flag count
df['flag_count'] = df[['flag_high_gt_diff', 'flag_high_pred_diff', 'flag_low_pose_eval', 
                       'flag_pred_gt_diff', 'flag_image_multiple_errors','flag_all_high_error_rate_image']].sum(axis=1)

# Print statistics for multiple error images
print(f"Images with multiple high errors: {len(images_with_multiple_high_errors)}")
print(f"Measurements from images with multiple high errors: {df['flag_image_multiple_errors'].sum()}")
if df['flag_image_multiple_errors'].sum() > 0:
    print(f"Percentage with this flag: {df['flag_image_multiple_errors'].mean()*100:.1f}%")

# Print detailed information about images with multiple high errors
print("\nDetailed information about images with multiple high errors:")
print("Image ID | High Error Count | Total Measurements | Error Rate (%)")
print("-" * 65)

for image in images_with_multiple_high_errors:
    high_error_count = high_error_counts_by_image[image]
    total_count = len(df[df['Label'] == image])
    error_rate = (high_error_count / total_count) * 100
    print(f"{image} | {high_error_count} | {total_count} | {error_rate:.1f}%")

# Group measurements by image and show the distribution
image_error_stats = pd.DataFrame({
    'Image': images_with_multiple_high_errors,
    'High_Error_Count': [high_error_counts_by_image[img] for img in images_with_multiple_high_errors],
    'Total_Count': [len(df[df['Label'] == img]) for img in images_with_multiple_high_errors]
})

image_error_stats['Error_Rate'] = (image_error_stats['High_Error_Count'] / image_error_stats['Total_Count']) * 100
image_error_stats = image_error_stats.sort_values('Error_Rate', ascending=False)

print("\nImages with multiple high errors sorted by error rate:")
print(image_error_stats.to_string(index=False))

# Update flag info display to include the new flag
df['flag_info'] = df.apply(lambda row: 
                        f"GT Diff: {row['flag_high_gt_diff']}, {(row['min_gt_diff']/row['expert_normalized_pixels']*100):.1f}%\n" +
                        f"Pred Diff: {row['flag_high_pred_diff']}, {(row['min_mape_pixels']):.1f}%\n" +
                        f"Pose Eval: {row['flag_low_pose_eval']}\n" +
                        f"Pred-GT Diff: {row['flag_pred_gt_diff']}, {(row['pred_pixel_gt_diff']/row['expert_normalized_pixels']*100):.1f}%\n" +
                        f"100% Error Rate Image: {row['flag_image_multiple_errors']}", 
                        axis=1)

# Print statistics on flags
print("\nFlag Statistics:")
print(f"High ground truth difference: {df['flag_high_gt_diff'].mean()*100:.1f}%")
print(f"High prediction difference: {df['flag_high_pred_diff'].mean()*100:.1f}%")
print(f"Low pose evaluation: {df['flag_low_pose_eval'].mean()*100:.1f}%")
print(f"High prediction ground truth difference: {df['flag_pred_gt_diff'].mean()*100:.1f}%")
# Calculate percentage of high errors that have each flag
high_error_df = df[df['high_error']]
print("\nPercentage of high errors (>10%) with each flag:")
print(f"High ground truth difference: {high_error_df['flag_high_gt_diff'].mean()*100:.1f}%")
print(f"High prediction difference: {high_error_df['flag_high_pred_diff'].mean()*100:.1f}%")
print(f"Low pose evaluation: {high_error_df['flag_low_pose_eval'].mean()*100:.1f}%")
print(f"High prediction ground truth difference: {high_error_df['flag_pred_gt_diff'].mean()*100:.1f}%")

# Calculate flag co-occurrence with high errors
print("\nPercentage of measurements with each flag that have high errors:")
if len(df[df['flag_high_gt_diff']]) > 0:
    print(f"High ground truth difference: {df[df['flag_high_gt_diff']]['high_error'].mean()*100:.1f}%")
if len(df[df['flag_high_pred_diff']]) > 0:
    print(f"High prediction difference: {df[df['flag_high_pred_diff']]['high_error'].mean()*100:.1f}%")
if len(df[df['flag_low_pose_eval']]) > 0:
    print(f"Low pose evaluation: {df[df['flag_low_pose_eval']]['high_error'].mean()*100:.1f}%")
if len(df[df['flag_pred_gt_diff']]) > 0:
    print(f"High prediction ground truth difference: {df[df['flag_pred_gt_diff']]['high_error'].mean()*100:.1f}%")

# Create visualization: Conditional bar chart showing percentage of high errors by flag
fig, ax = plt.subplots(figsize=(10, 6))

# For each flag, calculate percentage of high errors
flag_cols = ['flag_high_gt_diff',  'flag_high_pred_diff', 'flag_low_pose_eval', 'flag_pred_gt_diff']
flag_labels = ['High GT Diff\n(>20px)', 'High Pred Diff\n(>20px)', 'Low Pose Eval\n(<0.85)', 'High Pred GT Diff\n(>20px)']

flag_true_error_pcts = []
flag_false_error_pcts = []

for flag in flag_cols:
    # Percentage of high errors when flag is True
    if len(df[df[flag]]) > 0:
        flag_true_error_pcts.append(df[df[flag]]['high_error'].mean() * 100)
    else:
        flag_true_error_pcts.append(0)
    
    # Percentage of high errors when flag is False
    if len(df[~df[flag]]) > 0:
        flag_false_error_pcts.append(df[~df[flag]]['high_error'].mean() * 100)
    else:
        flag_false_error_pcts.append(0)

# Create bar chart
x = np.arange(len(flag_labels))
width = 0.35

ax.bar(x - width/2, flag_true_error_pcts, width, label='Flag Present', color='#e74c3c')
ax.bar(x + width/2, flag_false_error_pcts, width, label='Flag Absent', color='#2ecc71')

ax.set_xticks(x)
ax.set_xticklabels(flag_labels)
ax.set_ylabel('Percentage of Measurements with Error >10%')
ax.set_title('Impact of Error Flags on High Error Rate')
ax.legend()
ax.axhline(y=high_error_pct, color='gray', linestyle='--', label=f'Overall Error Rate ({high_error_pct:.1f}%)')
ax.set_ylim(0, 100)

# Add the overall rate to the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels)

plt.tight_layout()
plt.savefig('error_flags_impact.png', dpi=300)
plt.show()

# Create a heatmap showing co-occurrence of flags
flag_columns = ['flag_high_gt_diff', 'flag_high_pred_diff', 'flag_low_pose_eval', 'flag_pred_gt_diff']
flag_labels = ['High GT Diff', 'High Pred Diff', 'Low Pose Eval', 'High Pred-GT Diff']

# For the enhanced heatmap, add the multiple errors flag and MPE
enhanced_flag_columns = flag_columns + ['flag_image_multiple_errors', 'min_mpe']
enhanced_flag_labels = flag_labels + ['Multiple Errors in Image', 'MPE']

# Create a heatmap showing correlations between all variables including MPE
corr_matrix = pd.DataFrame()

# Add flag columns (binary variables)
for i, flag_col in enumerate(flag_columns + ['flag_image_multiple_errors']):
    corr_matrix[enhanced_flag_labels[i]] = df[flag_col].astype(float)

# Add MPE as a continuous variable
corr_matrix['MPE'] = df['min_mpe']

# Calculate correlation matrix
correlation_matrix = corr_matrix.corr()

# Plot the enhanced correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
plt.title('Correlation Matrix: Error Flags and MPE')
plt.tight_layout()
plt.show()

# Original heatmap showing co-occurrence of flags
heatmap_data = pd.DataFrame(index=flag_labels, columns=flag_labels)

for i, flag1 in enumerate(flag_columns):
    for j, flag2 in enumerate(flag_columns):
        if i == j:
            heatmap_data.iloc[i, j] = df[flag1].mean() * 100  # Percentage of measurements with this flag
        else:
            # Percentage of measurements with flag1 that also have flag2
            if df[flag1].sum() > 0:
                heatmap_data.iloc[i, j] = df[df[flag1]][flag2].astype(float).mean() * 100
            else:
                heatmap_data.iloc[i, j] = 0.0

# Convert to float to ensure compatibility with heatmap
heatmap_data = heatmap_data.astype(float)

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f', vmin=0, vmax=100)
plt.title('Flag Co-occurrence (%)')
plt.tight_layout()
plt.show()

# Create scatter plot with points colored by number of flags
fig = px.scatter(df, 
                x='min_gt_diff', 
                y='min_mpe',
                color='flag_count',
                color_continuous_scale='Viridis',
                hover_data=['PrawnID', 'Label', 'Pond_Type'],
                title='Error vs Ground Truth Difference, Colored by Number of Flags',
                labels={
                    'min_gt_diff': 'Ground Truth Pixel Difference',
                    'min_mpe': 'Min MPE (%)',
                    'flag_count': 'Number of Flags'
                })

# Add horizontal line at 10% error
fig.add_hline(y=10, line_dash="dash", line_color="red")
fig.update_layout(height=600, width=900)
fig.show()

# Create a more detailed scatter plot with flag information in hover
# Create a helper function to generate flag descriptions for hover text
def get_flag_descriptions(row):
    return (
        f"High GT Diff: {row['flag_high_gt_diff']}, val: {row['min_gt_diff']:.1f}px<br>" +
        f"High Pred Diff: {row['flag_high_pred_diff']}, val: {row['min_mape_pixels']:.1f}%<br>" +
        f"Low Pose Eval: {row['flag_low_pose_eval']}, val: {row['pose_eval_iou'] if 'pose_eval_iou' in row else row['pose_eval'] if 'pose_eval' in row else 'N/A'}<br>" +
        f"High Pred-GT Diff: {row['flag_pred_gt_diff']}, val: {row['pred_pixel_gt_diff']:.1f}px"
    )

# Add flag description column
df['flag_description'] = df.apply(get_flag_descriptions, axis=1)

# Create the enhanced scatter plot
fig = px.scatter(df, 
                x='min_gt_diff', 
                y='min_mpe',
                color='flag_count',
                color_continuous_scale='Viridis',
                hover_data={
                    'PrawnID': True,
                    'Label': True, 
                    'Pond_Type': True,
                    'min_gt_diff': ':.1f',
                    'min_mpe': ':.1f',
                    'flag_count': True,
                    'flag_description': True,
                    'pred_Distance_pixels': ':.1f',
                    'expert_normalized_pixels': ':.1f'
                },
                title='Error vs Ground Truth Difference, Colored by Number of Flags',
                labels={
                    'min_gt_diff': 'Ground Truth Pixel Difference',
                    'min_mpe': 'Min MPE (%)',
                    'flag_count': 'Number of Flags',
                    'flag_description': 'Flags',
                    'pred_Distance_pixels': 'Predicted Length (px)',
                    'expert_normalized_pixels': 'Best Expert Length (px)'
                })

# Add horizontal line at 10% error
fig.add_hline(y=10, line_dash="dash", line_color="red")
fig.update_layout(height=600, width=900)

# Customize hover template
fig.update_traces(
    hovertemplate='<b>ID:</b> %{customdata[0]}<br>' +
                 '<b>Image:</b> %{customdata[1]}<br>' +
                 '<b>GT Diff:</b> %{customdata[3]:.1f}px<br>' +
                 '<b>Error:</b> %{customdata[4]:.1f}%<br>' +
                 '<b>FLAGS:</b><br>%{customdata[6]}<br>' +
                 '<extra></extra>'
)

# Simplify the enhanced scatter plot with direct flags
# First create simple column flags for better hover display
df['flag_info'] = df.apply(lambda row: 
                        f"GT Diff: {row['flag_high_gt_diff']}, {row['min_gt_diff']:.1f}px\n" +
                        f"Pred Diff: {row['flag_high_pred_diff']}, {row['min_mape_pixels']:.1f}%\n" +
                        f"Pose Eval: {row['flag_low_pose_eval']}\n" +
                        f"Pred-GT Diff: {row['flag_pred_gt_diff']}, {row['pred_pixel_gt_diff']:.1f}px", 
                        axis=1)

# Create simple scatter plot with direct flag info
fig = px.scatter(df, 
                x='min_gt_diff', 
                y='min_mpe',
                color='flag_count',
                color_continuous_scale='Viridis',
                hover_data=['PrawnID', 'Label', 'flag_info'],
                title='Error vs Ground Truth Difference, Colored by Number of Flags',
                labels={
                    'min_gt_diff': 'Ground Truth Pixel Difference',
                    'min_mpe': 'Min MPE (%)',
                    'flag_count': 'Number of Flags'
                })

# Add horizontal line at 10% error
fig.add_hline(y=10, line_dash="dash", line_color="red")
fig.update_layout(height=600, width=900)

fig.show()

# Create categorical count plot showing distribution of flags in high error vs low error
fig = make_subplots(rows=1, cols=2, 
                   subplot_titles=("Measurements with Error ≤10%", "Measurements with Error >10%"),
                   column_widths=[0.5, 0.5])

# Count number of measurements with each flag count for low errors
low_error_counts = df[~df['high_error']]['flag_count'].value_counts().sort_index()
low_error_labels = [f"{count} flags" for count in low_error_counts.index]

# Count number of measurements with each flag count for high errors
high_error_counts = df[df['high_error']]['flag_count'].value_counts().sort_index()
high_error_labels = [f"{count} flags" for count in high_error_counts.index]

# Add bar charts
fig.add_trace(
    go.Bar(x=low_error_labels, y=low_error_counts.values, name="Error ≤10%", marker_color='#2ecc71'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=high_error_labels, y=high_error_counts.values, name="Error >10%", marker_color='#e74c3c'),
    row=1, col=2
)

fig.update_layout(height=500, width=900, 
                 title_text="Distribution of Flag Counts by Error Level",
                 showlegend=False)
fig.update_yaxes(title_text="Number of Measurements", row=1, col=1)
fig.update_yaxes(title_text="Number of Measurements", row=1, col=2)

fig.show()

# Create flag frequency bar chart with error rate overlay
flag_names = ['High GT Diff', 'High Expert Var', 'High Pred Diff', 'Low Pose Eval']
flag_counts = [df[flag].sum() for flag in flag_cols]
flag_high_error_pcts = [df[df[flag]]['high_error'].mean() * 100 if df[flag].sum() > 0 else 0 for flag in flag_cols]

# Create plotly bar chart
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add bars for flag counts
fig.add_trace(
    go.Bar(x=flag_names, y=flag_counts, name="Flag Count", marker_color='#3498db'),
    secondary_y=False
)

# Add line for error percentage
fig.add_trace(
    go.Scatter(x=flag_names, y=flag_high_error_pcts, name="% with Error >10%", 
               line=dict(color='#e74c3c', width=3), mode='lines+markers'),
    secondary_y=True
)

# Add horizontal line at overall error rate
fig.add_trace(
    go.Scatter(x=flag_names, y=[high_error_pct] * len(flag_names), name=f"Overall Error Rate ({high_error_pct:.1f}%)",
               line=dict(color='gray', width=1, dash='dash'), mode='lines'),
    secondary_y=True
)

fig.update_layout(
    title_text="Flag Frequency and Associated Error Rates",
    height=500, width=900
)

fig.update_yaxes(title_text="Number of Measurements", secondary_y=False)
fig.update_yaxes(title_text="Percentage with Error >10%", secondary_y=True, range=[0, 100])

fig.show()

# Keep only this scatter plot with pond type shapes
# Improve flag info display and use pond type for shapes
df['flag_info'] = df.apply(lambda row: 
                        f"GT Diff: {row['flag_high_gt_diff']}, {(row['min_gt_diff']/row['expert_normalized_pixels']*100):.1f}%\n" +
                        f"Pred Diff: {row['flag_high_pred_diff']}, {(row['min_mape_pixels']):.1f}%\n" +
                        f"Pose Eval: {row['flag_low_pose_eval']}\n" +
                        f"Pred-GT Diff: {row['flag_pred_gt_diff']}, {(row['pred_pixel_gt_diff']/row['expert_normalized_pixels']*100):.1f}%\n" +
                        f"100% Error Rate Image: {row['flag_image_multiple_errors']}", 
                        axis=1)

# Map pond types to marker symbols
pond_shapes = {
    'circle_male': 'circle', 
    'circle_female': 'diamond', 
    'square': 'square'
}

# Create figure using Plotly Graph Objects for more control
fig = go.Figure()

# Add traces for each pond type with improved hover information
for pond_type, shape in pond_shapes.items():
    pond_df = df[df['Pond_Type'] == pond_type]
    
    fig.add_trace(go.Scatter(
        x=pond_df['min_gt_diff'],
        y=pond_df['min_mpe'],
        mode='markers',
        marker=dict(
            size=10,
            symbol=shape,
            color=pond_df['flag_count'],
            colorscale='Viridis',
            colorbar=dict(title='Number of Flags') if pond_type == list(pond_shapes.keys())[0] else None,
            showscale=pond_type == list(pond_shapes.keys())[0]
        ),
        name=pond_type,
        text=pond_df['flag_info'],
        hovertemplate=
            "<b>ID:</b> %{customdata[0]}<br>" +
            "<b>Image:</b> %{customdata[1]}<br>" +
            "<b>Pond Type:</b> " + pond_type + "<br>" +
            "<b>Error (%):</b> %{customdata[2]:.1f}%<br>" +
            "<b>GT Diff (px):</b> %{customdata[3]:.1f}px<br>" +
            "<b>Flags:</b><br>%{text}<br>" +
            "<extra></extra>",
        customdata=pond_df[['PrawnID', 'Label', 'Pond_Type', 'min_mpe', 'min_gt_diff']].values
    ))

# Add horizontal line at 10% error threshold
fig.add_shape(
    type='line',
    x0=0, x1=df['min_gt_diff'].max() * 1.1,
    y0=10, y1=10,
    line=dict(color='red', dash='dash')
)

# Update layout
fig.update_layout(
    title='Error vs Ground Truth Difference, Colored by Number of Flags',
    xaxis_title='Ground Truth Pixel Difference',
    yaxis_title='Min MPE (%)',
    legend_title='Pond Type',
    height=600, width=900
)

fig.show()

# Create a new correlation subplot with two visualizations
from plotly.subplots import make_subplots

# Create a subplot with 1 row and 2 columns
correlation_fig = make_subplots(rows=1, cols=2, 
                                subplot_titles=("Error Distribution by Multiple Errors Flag", 
                                              "Percentage of High Errors by Flag"),
                                specs=[[{"type": "box"}, {"type": "bar"}]])

# 1. Box plot for error distribution by flag
correlation_fig.add_trace(
    go.Box(
        y=df[df['flag_image_multiple_errors']]['min_mpe'],
        name='Multiple Errors in Image',
        boxmean=True,
        marker_color='#e74c3c'
    ),
    row=1, col=1
)

# Create a more detailed box plot to show all errors by flag
box_fig = go.Figure()

# Add box plot for measurements from images with multiple errors
box_fig.add_trace(go.Box(
    y=df[df['flag_image_multiple_errors']]['min_mpe'],
    name='From Images with Multiple Errors',
    boxmean=True,
    marker_color='#e74c3c',
    boxpoints='all',  # Show all points
    jitter=0.3,
    pointpos=-1.5,
    hovertemplate=
        "<b>ID:</b> %{customdata[0]}<br>" +
        "<b>Image:</b> %{customdata[1]}<br>" +
        "<b>Error:</b> %{y:.1f}%<br>" +
        "<b>GT Diff:</b> %{customdata[2]:.1f}px<br>" +
        "<extra></extra>",
    customdata=df[df['flag_image_multiple_errors']][['PrawnID', 'Label', 'min_gt_diff']].values
))

# Add box plot for measurements from images without multiple errors
box_fig.add_trace(go.Box(
    y=df[~df['flag_image_multiple_errors']]['min_mpe'],
    name='From Images without Multiple Errors',
    boxmean=True,
    marker_color='#2ecc71',
    boxpoints='all',  # Show all points
    jitter=0.3,
    pointpos=-1.5,
    hovertemplate=
        "<b>ID:</b> %{customdata[0]}<br>" +
        "<b>Image:</b> %{customdata[1]}<br>" +
        "<b>Error:</b> %{y:.1f}%<br>" +
        "<b>GT Diff:</b> %{customdata[2]:.1f}px<br>" +
        "<extra></extra>",
    customdata=df[~df['flag_image_multiple_errors']][['PrawnID', 'Label', 'min_gt_diff']].values
))

# Add horizontal line at 10% error
box_fig.add_shape(
    type='line',
    x0=-0.5, x1=1.5,
    y0=10, y1=10,
    line=dict(color='red', dash='dash')
)

# Calculate statistics and add annotations
mean_with_flag = df[df['flag_image_multiple_errors']]['min_mpe'].mean()
mean_without_flag = df[~df['flag_image_multiple_errors']]['min_mpe'].mean()
median_with_flag = df[df['flag_image_multiple_errors']]['min_mpe'].median()
median_without_flag = df[~df['flag_image_multiple_errors']]['min_mpe'].median()

# Count points in each category
count_with_flag = len(df[df['flag_image_multiple_errors']])
count_without_flag = len(df[~df['flag_image_multiple_errors']])

# Count high errors in each category
high_with_flag = len(df[(df['flag_image_multiple_errors']) & (df['min_mpe'] > 10)])
high_without_flag = len(df[(~df['flag_image_multiple_errors']) & (df['min_mpe'] > 10)])

# Calculate percentages
pct_high_with_flag = (high_with_flag / count_with_flag) * 100
pct_high_without_flag = (high_without_flag / count_without_flag) * 100

# Add annotations for statistics
box_fig.add_annotation(
    x=0, y=max(50, df['min_mpe'].max() * 0.9),
    text=f"n={count_with_flag}<br>High Errors: {high_with_flag} ({pct_high_with_flag:.1f}%)<br>Mean: {mean_with_flag:.1f}%<br>Median: {median_with_flag:.1f}%",
    showarrow=False,
    align="center"
)

box_fig.add_annotation(
    x=1, y=max(50, df['min_mpe'].max() * 0.9),
    text=f"n={count_without_flag}<br>High Errors: {high_without_flag} ({pct_high_without_flag:.1f}%)<br>Mean: {mean_without_flag:.1f}%<br>Median: {median_without_flag:.1f}%",
    showarrow=False,
    align="center"
)

# Update layout
box_fig.update_layout(
    title='Error Distribution by Image Multiple Errors Flag',
    yaxis_title='Min MPE (%)',
    height=600, width=900,
    boxmode='group',
    # Use violin plot instead of box plot
    yaxis=dict(
        range=[0, max(50, df['min_mpe'].max() * 1.1)]
    )
)

box_fig.show()

# Create a scatter plot specifically for the Multiple Errors in Image flag
# Group points by image and calculate average min_mpe for each image
image_avg_error = df.groupby('Label')['min_mpe'].mean().reset_index()
image_error_count = df[df['min_mpe'] > 10].groupby('Label').size().reset_index(name='high_error_count')
image_data = pd.merge(image_avg_error, image_error_count, on='Label', how='left')
image_data['high_error_count'] = image_data['high_error_count'].fillna(0)
image_data['has_multiple_errors'] = image_data['high_error_count'] > 1

# Get count of measurements per image
image_measure_count = df.groupby('Label').size().reset_index(name='measurement_count')
image_data = pd.merge(image_data, image_measure_count, on='Label', how='left')

# Calculate error rate percentage
image_data['error_rate'] = (image_data['high_error_count'] / image_data['measurement_count']) * 100

# Create scatter plot of images
image_flag_fig = go.Figure()

# Add scatter plot for images with multiple high errors
image_flag_fig.add_trace(go.Scatter(
    x=image_data[image_data['has_multiple_errors']]['measurement_count'],
    y=image_data[image_data['has_multiple_errors']]['min_mpe'],
    mode='markers',
    marker=dict(
        size=image_data[image_data['has_multiple_errors']]['high_error_count'] * 5,
        color='red',
        opacity=0.7,
        line=dict(width=1, color='black')
    ),
    name='Multiple High Errors',
    text=image_data[image_data['has_multiple_errors']]['Label'],
    hovertemplate=
        "<b>Image:</b> %{text}<br>" +
        "<b>Total Measurements:</b> %{x}<br>" +
        "<b>Avg Error:</b> %{y:.1f}%<br>" +
        "<b>High Error Count:</b> %{marker.size/5}<br>" +
        "<b>Error Rate:</b> %{customdata:.1f}%<br>" +
        "<extra></extra>",
    customdata=image_data[image_data['has_multiple_errors']]['error_rate']
))

# Add scatter plot for images without multiple high errors
image_flag_fig.add_trace(go.Scatter(
    x=image_data[~image_data['has_multiple_errors']]['measurement_count'],
    y=image_data[~image_data['has_multiple_errors']]['min_mpe'],
    mode='markers',
    marker=dict(
        size=image_data[~image_data['has_multiple_errors']]['high_error_count'] * 5 + 5,
        color='green',
        opacity=0.7,
        line=dict(width=1, color='black')
    ),
    name='No Multiple High Errors',
    text=image_data[~image_data['has_multiple_errors']]['Label'],
    hovertemplate=
        "<b>Image:</b> %{text}<br>" +
        "<b>Total Measurements:</b> %{x}<br>" +
        "<b>Avg Error:</b> %{y:.1f}%<br>" +
        "<b>High Error Count:</b> %{marker.size/5 - 1}<br>" +
        "<b>Error Rate:</b> %{customdata:.1f}%<br>" +
        "<extra></extra>",
    customdata=image_data[~image_data['has_multiple_errors']]['error_rate']
))

# Add horizontal line at 10% error
image_flag_fig.add_shape(
    type='line',
    x0=0, x1=image_data['measurement_count'].max() * 1.1,
    y0=10, y1=10,
    line=dict(color='red', dash='dash')
)

# Update layout
image_flag_fig.update_layout(
    title='Image Error Analysis: Average Error vs Number of Measurements',
    xaxis_title='Number of Measurements in Image',
    yaxis_title='Average Error Rate (%)',
    height=600, width=900,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
    )
)

image_flag_fig.show()

# Create a more direct scatter plot showing relationship between MPE and image error density
# First, create a dictionary of image labels to high error counts
image_to_error_count = df[df['min_mpe'] > 10].groupby('Label').size().to_dict()

# Add a column for number of high errors in the same image
df['same_image_error_count'] = df['Label'].map(image_to_error_count).fillna(0)
# For each point, subtract 1 from count if the point itself has high error (to get count of OTHER errors)
df['other_high_errors_in_image'] = df.apply(
    lambda row: row['same_image_error_count'] - 1 if row['min_mpe'] > 10 else row['same_image_error_count'], 
    axis=1
)

# Create error density (high errors per measurement) for each image
total_measurements_per_image = df.groupby('Label').size().to_dict()
df['image_measurement_count'] = df['Label'].map(total_measurements_per_image)
df['image_error_density'] = df['same_image_error_count'] / df['image_measurement_count'] * 100

# Create scatter plot
error_density_fig = go.Figure()

# Add separate traces for high error and normal measurements
error_density_fig.add_trace(go.Scatter(
    x=df['other_high_errors_in_image'],
    y=df['min_mpe'],
    mode='markers',
    marker=dict(
        size=10,
        color=df['image_error_density'],
        colorscale='Viridis',
        colorbar=dict(title='Error Density (%)'),
        opacity=0.7,
        line=dict(width=1, color='black')
    ),
    text=df['Label'],
    hovertemplate=
        "<b>ID:</b> %{customdata[0]}<br>" +
        "<b>Image:</b> %{text}<br>" +
        "<b>Error:</b> %{y:.1f}%<br>" +
        "<b>Other High Errors:</b> %{x}<br>" + 
        "<b>Total Measurements:</b> %{customdata[1]}<br>" +
        "<b>Error Density:</b> %{marker.color:.1f}%<br>" +
        "<extra></extra>",
    customdata=df[['PrawnID', 'image_measurement_count']].values
))

# Add horizontal line at 10% error
error_density_fig.add_shape(
    type='line',
    x0=-0.5, x1=df['other_high_errors_in_image'].max() + 0.5,
    y0=10, y1=10,
    line=dict(color='red', dash='dash')
)

# Add a vertical line at x=0 to separate points with no other high errors
error_density_fig.add_shape(
    type='line',
    x0=0, x1=0,
    y0=0, y1=df['min_mpe'].max() * 1.1,
    line=dict(color='gray', dash='dash')
)

# Update layout
error_density_fig.update_layout(
    title='Measurement Error vs Number of Other High Errors in Same Image',
    xaxis_title='Number of Other High Errors in Same Image',
    yaxis_title='Min MPE (%)',
    height=600, width=900,
    xaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=1
    )
)

error_density_fig.show()

# Create categorical scatter plot - MPE vs categorical variables
# Create category based on number of other high errors in the same image
df['error_category'] = pd.cut(
    df['other_high_errors_in_image'],
    bins=[-1, 0, 1, 2, float('inf')],
    labels=['0', '1', '2', '3+']
)

# Create the categorical scatter plot - REMOVING THIS TO FIX ERROR
# cat_error_fig = go.Figure()
# ... rest of cat_error_fig code ...

# Create another categorical plot by pond type - REMOVING THIS TO FIX ERROR
# pond_error_fig = go.Figure()
# ... rest of pond_error_fig code ...

# Create a simple correlation plot between MPE and multiple errors flag
# Calculate the correlation coefficient
correlation = df['min_mpe'].corr(df['flag_image_multiple_errors'])

# Create a simplified x-axis (0 = no multiple errors, 1 = multiple errors)
x_values = df['flag_image_multiple_errors'].astype(int)

# Create figure
corr_fig = go.Figure()

# Add scatter plot
corr_fig.add_trace(go.Scatter(
    x=x_values,
    y=df['min_mpe'],
    mode='markers',
    marker=dict(
        size=8,
        opacity=0.7,
        color=df['min_mpe'],
        colorscale='Viridis',
        colorbar=dict(title='Min MPE (%)'),
    ),
    text=df['Label'],
    hovertemplate=
        "<b>ID:</b> %{customdata[0]}<br>" +
        "<b>Image:</b> %{text}<br>" +
        "<b>Error:</b> %{y:.1f}%<br>" +
        "<b>Multiple Errors in Image:</b> %{x}<br>" +
        "<extra></extra>",
    customdata=df[['PrawnID']].values
))

# Add linear regression line
x_range = [0, 1]
slope, intercept = np.polyfit(x_values, df['min_mpe'], 1)
corr_fig.add_trace(go.Scatter(
    x=x_range,
    y=[intercept + slope * x for x in x_range],
    mode='lines',
    line=dict(color='red', width=2, dash='dash'),
    name=f'Correlation: {correlation:.3f}'
))

# Add horizontal line at 10% error
corr_fig.add_shape(
    type='line',
    x0=-0.1, x1=1.1,
    y0=10, y1=10,
    line=dict(color='red', dash='dot')
)

# Calculate mean for each group
mean_no_multiple = df[~df['flag_image_multiple_errors']]['min_mpe'].mean()
mean_with_multiple = df[df['flag_image_multiple_errors']]['min_mpe'].mean()

# Add text annotations for means
corr_fig.add_annotation(
    x=0,
    y=mean_no_multiple,
    text=f"Mean: {mean_no_multiple:.1f}%",
    showarrow=True,
    arrowhead=1,
    ax=40,
    ay=0
)

corr_fig.add_annotation(
    x=1,
    y=mean_with_multiple,
    text=f"Mean: {mean_with_multiple:.1f}%",
    showarrow=True,
    arrowhead=1,
    ax=-40,
    ay=0
)

# Update layout
corr_fig.update_layout(
    title=f'Correlation Between MPE and 100% Error Rate Images (r = {correlation:.3f})',
    xaxis_title='100% Error Rate Image (0=No, 1=Yes)',
    yaxis_title='Min MPE (%)',
    height=500, width=800,
    xaxis=dict(
        tickmode='array',
        tickvals=[0, 1],
        ticktext=['No 100% Error Rate', '100% Error Rate']
    )
)

corr_fig.show()

# Now analyze if measurements with the new flag (100% Error Rate) are more likely to have high errors
mean_no_multiple = df[~df['flag_image_multiple_errors']]['min_mpe'].mean()
mean_with_multiple = df[df['flag_image_multiple_errors']]['min_mpe'].mean()

# Print statistics
print(f"\nMean MPE for measurements from 100% error rate images: {mean_with_multiple:.2f}%")
print(f"Mean MPE for other measurements: {mean_no_multiple:.2f}%")

# Create a comprehensive box plot showing all error flag types
all_flags_fig = go.Figure()

# Define all the flag columns and nice display names
flag_columns = [
    'flag_high_gt_diff',
    'flag_high_pred_diff',
    'flag_low_pose_eval',
    'flag_pred_gt_diff',
    'flag_image_multiple_errors'
]

flag_names = [
    'GT-Expert pixel  diff >10%',
    'Prediction-Expert pixel diff >10%',
    'Pose error >15%',
    'Prediction-GT pixel diff >10%',
    '100% Error Rate Image'
]

descriptive_tooltips = [
    'High difference between ground truth and expert measurements (>10% of size)',
    'High difference between prediction and expert measurements (>10% of size)',
    'Low pose evaluation score indicating poor keypoint detection (<0.85)',
    'High difference between ground truth and prediction (>10% of size)',
    'Image contains multiple measurements with high errors'
]

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

# Add a box plot for each flag (True and False values)
for i, (flag_col, flag_name, tooltip) in enumerate(zip(flag_columns, flag_names, descriptive_tooltips)):
    # Add box plot for when flag is True
    all_flags_fig.add_trace(go.Box(
        y=df[df[flag_col]]['min_mpe'],
        name=flag_name + ' (Yes)',
        boxmean=True,
        marker_color=colors[i],
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.5,
        hovertemplate=
            f"<b>{flag_name}</b><br>" +
            f"<i>{tooltip}</i><br>" +
            "<b>ID:</b> %{customdata[0]}<br>" +
            "<b>Image:</b> %{customdata[1]}<br>" +
            "<b>Error:</b> %{y:.1f}%<br>" +
            "<b>GT Diff:</b> %{customdata[2]:.1f}px<br>" +
            "<extra></extra>",
        customdata=df[df[flag_col]][['PrawnID', 'Label', 'min_gt_diff']].values
    ))
    
    # Add box plot for when flag is False
    all_flags_fig.add_trace(go.Box(
        y=df[~df[flag_col]]['min_mpe'],
        name=flag_name + ' (No)',
        boxmean=True,
        marker_color=colors[i],
        opacity=0.5,
        boxpoints=False,  # Don't show individual points for False values (cleaner)
        hovertemplate=
            f"<b>{flag_name} (Absent)</b><br>" +
            f"<i>{tooltip}</i><br>" +
            "<b>Mean:</b> %{mean:.1f}%<br>" +
            "<b>Median:</b> %{median:.1f}%<br>" +
            "<extra></extra>"
    ))

# Add horizontal line at 10% error
all_flags_fig.add_shape(
    type='line',
    x0=-0.5, x1=len(flag_columns)*2 - 0.5,
    y0=10, y1=10,
    line=dict(color='red', dash='dash')
)

# Update layout with wider boxes
all_flags_fig.update_layout(
    title='Error Distribution by Flag Type',
    yaxis_title='Min MPE (%)',
    height=700, width=1200,
    boxmode='group',
    yaxis=dict(
        range=[0, max(50, df['min_mpe'].max() * 1.1)]
    ),
    # Increase box width by adjusting gaps
    boxgroupgap=0.2,  # Gap between different flag types (smaller = more space for boxes)
    boxgap=0.3,       # Gap between Yes/No boxes within a group (larger = wider boxes)
    margin=dict(l=50, r=50, t=80, b=80)  # Add more margin for better visibility
)

# Update the individual traces to make them wider
for i in range(len(all_flags_fig.data)):
    all_flags_fig.data[i].update(
        width=0.6,  # Explicitly set box width
        quartilemethod="linear"  # This can make the boxes look more substantial
    )

all_flags_fig.show()

# Add analysis to determine the primary cause of errors when multiple flags are present
# First, calculate correlation between each flag and error percentage
flag_correlations = []
for flag_col in flag_columns:
    correlation = df[flag_col].corr(df['min_mpe'])
    flag_correlations.append(correlation)

# Create a DataFrame to store flag correlations
correlation_df = pd.DataFrame({
    'Flag': flag_names,
    'Correlation': flag_correlations
})
correlation_df = correlation_df.sort_values('Correlation', ascending=False)

print("\nCorrelation between flags and error percentage:")
for flag_col, flag_name in zip(
    ['flag_low_pose_eval', 'flag_high_pred_diff', 'flag_pred_gt_diff', 'flag_image_multiple_errors', 'flag_high_gt_diff'],
    ['Pose error >15%', 'Prediction-Expert pixel diff >10%', 'Prediction-GT pixel diff >10%', 'Multiple Errors in Same Image', 'GT-Expert pixel  diff >10%']
):
    correlation = df['min_mpe'].corr(df[flag_col])
    print(f"{flag_name}: {correlation:.3f}")

# Create a function to identify the most likely cause flag for each measurement
def get_primary_flag(row):
    if row['flag_count'] == 0:
        return "No Flags"
    
    # Get all active flags and their correlations
    active_flags = []
    for flag_col, flag_name in zip(flag_columns, flag_names):
        if row[flag_col]:
            corr = correlation_df[correlation_df['Flag'] == flag_name]['Correlation'].values[0]
            active_flags.append((flag_name, corr))
    
    # Sort by correlation strength
    active_flags.sort(key=lambda x: x[1], reverse=True)
    
    return active_flags[0][0] if active_flags else "No Flags"

# Add a column for primary cause
df['primary_cause'] = df.apply(get_primary_flag, axis=1)

# Create a visualization of error rates by primary cause
primary_cause_fig = go.Figure()

# Group data by primary cause
for cause in df['primary_cause'].unique():
    if cause == "No Flags":
        continue  # Skip measurements with no flags
        
    cause_df = df[df['primary_cause'] == cause]
    
    primary_cause_fig.add_trace(go.Box(
        y=cause_df['min_mpe'],
        name=cause,
        boxmean=True,
        marker_color=colors[flag_names.index(cause) if cause in flag_names else -1],
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.5,
        hovertemplate=
            f"<b>Primary Cause: {cause}</b><br>" +
            "<b>ID:</b> %{customdata[0]}<br>" +
            "<b>Image:</b> %{customdata[1]}<br>" +
            "<b>Error:</b> %{y:.1f}%<br>" +
            "<b>Flag Count:</b> %{customdata[2]}<br>" +
            "<extra></extra>",
        customdata=cause_df[['PrawnID', 'Label', 'flag_count']].values
    ))

# Add horizontal line at 10% error
primary_cause_fig.add_shape(
    type='line',
    x0=-0.5, x1=len(df['primary_cause'].unique()),
    y0=10, y1=10,
    line=dict(color='red', dash='dash')
)

# Update layout
primary_cause_fig.update_layout(
    title='Error Distribution by Primary Cause',
    yaxis_title='Min MPE (%)',
    height=600, width=1000,
    boxmode='group',
    yaxis=dict(
        range=[0, max(50, df['min_mpe'].max() * 1.1)]
    ),
    margin=dict(l=50, r=50, t=80, b=120)  # Add more margin for better visibility
)

# Update the individual traces to make them wider
for i in range(len(primary_cause_fig.data)):
    primary_cause_fig.data[i].update(
        width=0.6,  # Explicitly set box width
        quartilemethod="linear"  # This can make the boxes look more substantial
    )

primary_cause_fig.show()

# Also create a scatter plot to visualize measurements with multiple flags
multi_flag_fig = px.scatter(
    df[df['flag_count'] > 1],  # Only include measurements with multiple flags
    x='flag_count',
    y='min_mpe',
    color='primary_cause',
    size='min_gt_diff',  # Size points by ground truth difference
    hover_data=['PrawnID', 'Label', 'primary_cause', 'flag_info'],
    title='Measurements with Multiple Flags: Which Flag is Most Correlated with Errors?',
    labels={
        'flag_count': 'Number of Flags',
        'min_mpe': 'Error (%)',
        'primary_cause': 'Primary Cause',
        'min_gt_diff': 'GT Diff (px)'
    }
)

# Add horizontal line at 10% error
multi_flag_fig.add_hline(y=10, line_dash="dash", line_color="red")

# Update layout
multi_flag_fig.update_layout(
    height=600, width=1000
)

multi_flag_fig.show()

# Create a new comprehensive box plot with mutually exclusive categories
exclusive_flags_fig = go.Figure()

# Define priorities based on percentage thresholds
# New approach: Add columns with the actual percentage values for each flag
df['gt_diff_pct'] = df['min_gt_diff'] / df['expert_normalized_pixels'] * 100
df['pred_diff_pct'] = df['min_mape_pixels']
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
    
    
    if row['pose_pct'] > 15:
        return 'Pose error >15%'

    if row['flag_all_high_error_rate_image']:
      print("All High error rate image")
      return 'All High error rate image'
    
    if row['flag_all_high_gt_expert_error']:
        return 'All GT-Expert error >10%'

    if row['flag_all_high_pixel_error']:
        return 'All High Pixel Error'
    
  
    if row['flag_pred_gt_diff']:
        return 'Prediction-GT pixel diff >3%'


    # Check for multiple errors last
    if row['flag_image_multiple_errors']:
        return 'Multiple Errors in Same Image'
    
    # If no flags at all
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

# Add a box plot for each exclusive category
categories = [cat for cat in priority_names] + ['No Flags']
for i, category in enumerate(categories):
    cat_df = df[df['assigned_category'] == category]
    
    if len(cat_df) > 0:  # Only add if there are measurements in this category
        color = priority_colors[i] if i < len(priority_colors) else 'gray'
        
        exclusive_flags_fig.add_trace(go.Box(
            y=cat_df['min_mpe'],
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
                "<b>Error:</b> %{y:.1f}%<br>" +
                "<b>GT Diff:</b> %{customdata[2]:.1f}px (%{customdata[4]:.1f}%)<br>" +
                "<b>Pred-GT Diff:</b> %{customdata[5]:.1f}px (%{customdata[6]:.1f}%)<br>" +
                "<b>Pred Diff:</b> %{customdata[7]:.1f}px (%{customdata[8]:.1f}%)<br>" +
                "<b>Flag Count:</b> %{customdata[3]}<br>" +
                "<b>best pixels</b> %{customdata[10]:.1f}px<br>" +
                "<b>pred pixels</b> %{customdata[11]:.1f}px<br>" +
                "<b>pixel diff 1</b> %{customdata[12]:.1f}%<br>" +
                "<b>pixel diff 2</b> %{customdata[13]:.1f}%<br>" +
                "<b>pixel diff 3</b> %{customdata[14]:.1f}%<br>" ,
            customdata=cat_df[['PrawnID', 'Label', 'min_gt_diff', 'flag_count', 
                              'gt_diff_pct', 'pred_pixel_gt_diff', 'pred_gt_diff_pct',
                            'min_error_pixels', 'min_mape_pixels','Pond_Type','best_length_pixels','pred_Distance_pixels','pixel_diff_pct_1','pixel_diff_pct_2','pixel_diff_pct_3']].values
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

# Focus on images with 100% error rate
perfect_error_images = images_with_multiple_high_errors
print("\n---------- IMAGES WITH MULTIPLE HIGH ERRORS ----------")
print(f"Number of images with multiple high errors: {len(perfect_error_images)}")
print("\nDetailed information about images with multiple high errors:")
print("Image ID | High Error Count | Total Measurements | Error Rate (%)")
print("-" * 65)

for image in perfect_error_images:
    high_error_count = high_error_counts_by_image[image]
    total_count = len(df[df['Label'] == image])
    error_rate = (high_error_count / total_count) * 100
    print(f"{image} | {high_error_count} | {total_count} | {error_rate:.1f}%")

# Get all measurements from these images
perfect_error_df = df[df['Label'].isin(perfect_error_images)]

# Analyze what other flags are present in these images
flag_columns_display = ['flag_high_gt_diff', 'flag_high_pred_diff', 'flag_low_pose_eval', 'flag_pred_gt_diff']
flag_names_display = [
    'GT-Expert Diff >10%',
    'Prediction-Expert Diff >10%',
    'Pose Eval <0.85',
    'GT-Prediction Diff >10%',
    'Multiple Errors in Same Image'
]

print("\nFlag distribution in images with multiple high errors:")
for flag_col, flag_name in zip(flag_columns_display, flag_names_display):
    percentage = perfect_error_df[flag_col].mean() * 100
    count = perfect_error_df[flag_col].sum()
    total = len(perfect_error_df)
    print(f"{flag_name}: {count}/{total} measurements ({percentage:.1f}%)")

# Show the average MPE in these images
print(f"\nAverage MPE in images with multiple high errors: {perfect_error_df['min_mpe'].mean():.2f}%")
print(f"Min MPE: {perfect_error_df['min_mpe'].min():.2f}%, Max MPE: {perfect_error_df['min_mpe'].max():.2f}%")

# Show the pond type distribution for these images
pond_distribution = perfect_error_df['Pond_Type'].value_counts()
print("\nPond type distribution in images with multiple high errors:")
for pond_type, count in pond_distribution.items():
    percentage = count / len(perfect_error_df) * 100
    print(f"{pond_type}: {count} measurements ({percentage:.1f}%)")

# Create a visual representation of perfect error images without using flag_info
perfect_error_fig = px.scatter(
    perfect_error_df,
    x="min_gt_diff", 
    y="min_mpe",
    color="Pond_Type", 
    symbol="Pond_Type",
    size="flag_count",
    hover_data=["PrawnID", "Label"],
    title="Measurements from Images with Multiple High Errors",
    labels={
        "min_gt_diff": "Ground Truth Difference (px)",
        "min_mpe": "Error (%)",
        "flag_count": "Number of Flags"
    }
)

# Add horizontal line at 10% error
perfect_error_fig.add_hline(y=10, line_dash="dash", line_color="red")

# Update layout
perfect_error_fig.update_layout(height=600, width=900)
perfect_error_fig.show()

# ADDITIONAL ANALYSIS: Remove measurements above 10% MPE and recalculate metrics
print("\n\n====================== FILTERED ANALYSIS ======================")
print("Removing all measurements with MPE > 10% and recalculating metrics")

# Create filtered dataset
filtered_df = df[df['min_mpe'] <= 10]
print(f"\nOriginal dataset: {len(df)} measurements")
print(f"Filtered dataset: {len(filtered_df)} measurements")
print(f"Removed: {len(df) - len(filtered_df)} measurements ({((len(df) - len(filtered_df)) / len(df) * 100):.1f}% of data)")

# Calculate error metrics for both datasets
print("\nError Metrics Comparison:")
print("Metric | All Data | Filtered Data (<=10% MPE)")
print("-" * 50)

# Mean Percentage Error (MPE)
mpe_all = df['min_mpe'].mean()
mpe_filtered = filtered_df['min_mpe'].mean()
print(f"Mean MPE | {mpe_all:.2f}% | {mpe_filtered:.2f}%")

# Mean Absolute Error (MAE) in pixels
# Calculate actual pixel error
df['pixel_error'] = df['min_gt_diff']  # Ground truth difference in pixels
filtered_df['pixel_error'] = filtered_df['min_gt_diff']  # Ground truth difference in pixels

mae_px_all = df['pixel_error'].mean()
mae_px_filtered = filtered_df['pixel_error'].mean()
print(f"MAE (pixels) | {mae_px_all:.2f}px | {mae_px_filtered:.2f}px")

# Calculate MAE as percentage of prawn size
df['pixel_error_pct'] = df['pixel_error'] / df['expert_normalized_pixels'] * 100
filtered_df['pixel_error_pct'] = filtered_df['pixel_error'] / filtered_df['expert_normalized_pixels'] * 100

mae_pct_all = df['pixel_error_pct'].mean()
mae_pct_filtered = filtered_df['pixel_error_pct'].mean()
print(f"MAE (% of size) | {mae_pct_all:.2f}% | {mae_pct_filtered:.2f}%")

# Flag distribution in filtered dataset
print("\nFlag distribution in filtered dataset (≤10% MPE):")
for flag_col, flag_name in zip(flag_columns_display, flag_names_display):
    percentage = filtered_df[flag_col].mean() * 100
    count = filtered_df[flag_col].sum()
    total = len(filtered_df)
    print(f"{flag_name}: {count}/{total} measurements ({percentage:.1f}%)")

# Pond type distribution in filtered dataset
print("\nPond type distribution in filtered dataset:")
pond_filtered_distribution = filtered_df['Pond_Type'].value_counts()
for pond_type, count in pond_filtered_distribution.items():
    percentage = count / len(filtered_df) * 100
    print(f"{pond_type}: {count} measurements ({percentage:.1f}%)")

# Create a comparison visualization
comparison_fig = px.histogram(
    df, 
    x="min_mpe",
    color=df["min_mpe"] > 10,
    barmode="overlay",
    nbins=40,
    opacity=0.7,
    color_discrete_map={False: "green", True: "red"},
    labels={"min_mpe": "Min MPE (%)", "color": "MPE > 10%"},
    title="Distribution of Error Rates (Red = Removed in Filtered Analysis)"
)

# Add vertical line at 10%
comparison_fig.add_vline(x=10, line_dash="dash", line_color="black")
comparison_fig.update_layout(height=500, width=900)
comparison_fig.show()

# Calculate MAE - abs(Length_fov(mm) - best_length)
mae_all = abs(df['Length_fov(mm)'] - df['best_length']).mean()
mae_filtered = abs(filtered_df['Length_fov(mm)'] - filtered_df['best_length']).mean()
print(f"MAE (mm) | {mae_all:.2f}mm | {mae_filtered:.2f}mm")

