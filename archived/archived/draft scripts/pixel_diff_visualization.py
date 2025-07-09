import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

# Calculate minimum percentage error (MPE) if not already in dataframe
if 'min_mpe' not in df.columns:
    df['min_mpe'] = df[['MPError_fov_min', 'MPError_fov_max', 'MPError_fov_median']].min(axis=1)

# Calculate pixel differences
df['pred_pixels_diff_1'] = abs(df['pred_Distance_pixels'] - df['Length_1_pixels'])
df['pred_pixels_diff_2'] = abs(df['pred_Distance_pixels'] - df['Length_2_pixels'])
df['pred_pixels_diff_3'] = abs(df['pred_Distance_pixels'] - df['Length_3_pixels'])

# Calculate minimum pixel difference
df['min_pixel_diff'] = df[['pred_pixels_diff_1', 'pred_pixels_diff_2', 'pred_pixels_diff_3']].min(axis=1)

# Print basic statistics
print(f"Minimum pixel difference: {df['min_pixel_diff'].min():.2f}")
print(f"Maximum pixel difference: {df['min_pixel_diff'].max():.2f}")
print(f"Mean pixel difference: {df['min_pixel_diff'].mean():.2f}")
print(f"Median pixel difference: {df['min_pixel_diff'].median():.2f}")

# Define color categories for pixel differences
pixel_diff_bins = [0, 5, 10, 20, float('inf')]
pixel_diff_labels = ['<5px', '5-10px', '10-20px', '>20px']
pixel_diff_colors = {
    '<5px': '#2ecc71',    # Green
    '5-10px': '#f1c40f',  # Yellow
    '10-20px': '#e74c3c', # Red
    '>20px': '#9b59b6'    # Purple
}

# Create separate figure for each pond type
for pond_type in df['Pond_Type'].unique():
    # Filter data for this pond type
    pond_df = df[df['Pond_Type'] == pond_type].copy()
    
    # Categorize pixel differences
    pond_df['pixel_diff_category'] = pd.cut(
        pond_df['min_pixel_diff'],
        bins=pixel_diff_bins,
        labels=pixel_diff_labels
    )
    
    # Sort by min MPE for consistent visualization
    pond_df = pond_df.sort_values(by='min_mpe')
    
    # Create figure
    fig = go.Figure()
    
    # Create y-values to spread points vertically
    y_values = np.linspace(0.1, 0.9, len(pond_df))
    
    # Create scatter plot for each pixel difference category
    for category in pixel_diff_labels:
        cat_df = pond_df[pond_df['pixel_diff_category'] == category]
        if len(cat_df) > 0:
            # Get indices for y values
            indices = [list(pond_df.index).index(idx) for idx in cat_df.index]
            cat_y_values = [y_values[i] for i in indices]
            
            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=cat_df['min_mpe'],
                    y=cat_y_values,
                    mode='markers',
                    marker=dict(
                        color=pixel_diff_colors[category],
                        size=10,
                        line=dict(width=1, color='black')
                    ),
                    name=category,
                    hovertemplate=
                    "ID: %{customdata[0]}<br>" + 
                    "Image: %{customdata[1]}<br>" +
                    "Error (%): %{x:.1f}%<br>" + 
                    "Pixel Diff: %{customdata[2]:.1f}px<br>" +
                    "<extra></extra>",
                    customdata=cat_df[['PrawnID', 'Label', 'min_pixel_diff']].values
                )
            )
    
    # Add vertical lines at 5% and 10% error boundaries
    fig.add_shape(
        type="line", x0=5, x1=5, y0=0, y1=1,
        line=dict(color="gray", width=1, dash="dash")
    )
    fig.add_shape(
        type="line", x0=10, x1=10, y0=0, y1=1,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Add annotations
    fig.add_annotation(x=5, y=0.95, text="5%", showarrow=False)
    fig.add_annotation(x=10, y=0.95, text="10%", showarrow=False)

    # Update layout
    fig.update_layout(
        height=500,
        width=800,
        title_text=f"Error Distribution by Pixel Difference - {pond_type} Pond",
        template="plotly_white",
        xaxis_title="Min MPE (%)",
        yaxis_title="",
        yaxis_showticklabels=False,
        xaxis_range=[0, df['min_mpe'].max() * 1.1],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    # Display the figure
    fig.show()

# Print statistics by pond type
print("\nPixel difference statistics by pond type:")
for pond_type in df['Pond_Type'].unique():
    pond_df = df[df['Pond_Type'] == pond_type]
    print(f"\n{pond_type} pond:")
    print(f"  Min: {pond_df['min_pixel_diff'].min():.2f}px")
    print(f"  Max: {pond_df['min_pixel_diff'].max():.2f}px")
    print(f"  Mean: {pond_df['min_pixel_diff'].mean():.2f}px")
    print(f"  Median: {pond_df['min_pixel_diff'].median():.2f}px")
    
    # Distribution of points across pixel difference categories
    pixel_categories = pd.cut(
        pond_df['min_pixel_diff'],
        bins=pixel_diff_bins,
        labels=pixel_diff_labels
    ).value_counts()
    
    print("  Distribution:")
    for category, count in pixel_categories.items():
        percentage = count / len(pond_df) * 100
        print(f"    {category}: {count} points ({percentage:.1f}%)") 