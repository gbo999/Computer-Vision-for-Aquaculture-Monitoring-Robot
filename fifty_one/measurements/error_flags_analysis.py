import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# Get the column name that has the minimum MPE for each row
min_mpe_column = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].idxmin(axis=1)

# Map to the correct index (1, 2, or 3)
min_mpe_index = min_mpe_column.map(column_to_index)

# Now we can use this index to get the corresponding length
df['best_length'] = df.apply(lambda row: row[f'Length_{min_mpe_index[row.name]}'], axis=1)

# Calculate best length in pixels using the corresponding scale
df['best_length_pixels'] = df.apply(lambda row: row[f'Length_{min_mpe_index[row.name]}_pixels'], axis=1)

# # Calculate pixel differences
# df['min_expert_pixels'] = df[['Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].min(axis=1)
# df['max_expert_pixels'] = df[['Length_1_pixels', 'Length_2_pixels', 'Length_3_pixels']].max(axis=1)
# df['expert_range_pixels'] = df['max_expert_pixels'] - df['min_expert_pixels']

df['pred_pixels_diff'] = abs(df['pred_Distance_pixels'] - df['best_length_pixels'])

df['pred_pixel_gt_diff'] = abs(df['Length_ground_truth_annotation_pixels'] - df['pred_Distance_pixels'])

# Flag measurements with errors > 10%
df['high_error'] = df['min_mpe'] > 10

# Calculate proportion of high errors
high_error_pct = df['high_error'].mean() * 100
print(f"Percentage of measurements with errors > 10%: {high_error_pct:.1f}%")



df['min_gt_diff'] = abs(df['best_length_pixels'] - df['Length_ground_truth_annotation_pixels'])







# Define flags for potential error sources
# 1. High pixel difference between ground truth and expert
df['flag_high_gt_diff'] = df['min_gt_diff']/df['best_length_pixels']*100 > 10

# # 2. High variability between expert measurements
# df['flag_high_expert_var'] = df['expert_range_pixels'] > 15

# 3. High pixel difference between prediction and expert 
df['flag_high_pred_diff'] = df['pred_pixels_diff'] / df['best_length_pixels'] * 100 > 10

# 4. Low pose evaluation score (if available)
if 'pose_eval' in df.columns:
    df['flag_low_pose_eval'] = df['pose_eval'] < 0.85
elif 'pose_eval_iou' in df.columns:
    df['flag_low_pose_eval'] = df['pose_eval_iou'] < 0.85
else:
    df['flag_low_pose_eval'] = False
    print("Warning: No pose evaluation column found!")

df['flag_pred_gt_diff'] = df['pred_pixel_gt_diff']/df['best_length_pixels']*100 > 10

# Count how many flags each measurement has
df['flag_count'] = df[['flag_high_gt_diff',  'flag_high_pred_diff', 'flag_low_pose_eval', 'flag_pred_gt_diff']].sum(axis=1)





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
heatmap_data = pd.DataFrame(index=flag_labels, columns=flag_labels)

for i, flag1 in enumerate(flag_cols):
    for j, flag2 in enumerate(flag_cols):
        # Calculate conditional probability: P(flag2=True | flag1=True)
        if len(df[df[flag1]]) > 0:
            # Convert to numeric values explicitly
            heatmap_data.iloc[i, j] = df[df[flag1]][flag2].astype(float).mean() * 100
        else:
            heatmap_data.iloc[i, j] = 0.0

# Convert the entire DataFrame to float type
heatmap_data = heatmap_data.astype(float)

plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f', vmin=0, vmax=100)
plt.title('Co-occurrence of Error Flags (% of rows with row flag that also have column flag)')
plt.tight_layout()
plt.savefig('error_flags_cooccurrence.png', dpi=300)
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
        f"High Pred Diff: {row['flag_high_pred_diff']}, val: {row['pred_pixels_diff']:.1f}px<br>" +
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
                    'best_length_pixels': ':.1f'
                },
                title='Error vs Ground Truth Difference, Colored by Number of Flags',
                labels={
                    'min_gt_diff': 'Ground Truth Pixel Difference',
                    'min_mpe': 'Min MPE (%)',
                    'flag_count': 'Number of Flags',
                    'flag_description': 'Flags',
                    'pred_Distance_pixels': 'Predicted Length (px)',
                    'best_length_pixels': 'Best Expert Length (px)'
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
                        f"Pred Diff: {row['flag_high_pred_diff']}, {row['pred_pixels_diff']:.1f}px\n" +
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
                        f"GT Diff: {row['flag_high_gt_diff']}, {(row['min_gt_diff']/row['best_length_pixels']*100):.1f}%\n" +
                        f"Pred Diff: {row['flag_high_pred_diff']}, {(row['pred_pixels_diff']/row['best_length_pixels']*100):.1f}%\n" +
                        f"Pose Eval: {row['flag_low_pose_eval']}\n" +
                        f"Pred-GT Diff: {row['flag_pred_gt_diff']}, {(row['pred_pixel_gt_diff']/row['best_length_pixels']*100):.1f}%", 
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
            "<b>Pond Type:</b> %{customdata[2]}<br>" +
            "<b>Error (%):</b> %{customdata[3]:.1f}%<br>" +
            "<b>GT Diff (px):</b> %{customdata[4]:.1f}px<br>" +
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