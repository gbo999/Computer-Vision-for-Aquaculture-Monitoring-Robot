import pandas as pd
import plotly.express as px
import numpy as np

# Expected values
expected_big_total = 180  # 18cm in mm
expected_small_total = 145  # 14.5cm in mm

# Load the CSV file
df = pd.read_csv('runs/pose/predict80/length_analysis_new.csv')

# Create a new column for pond type
df['pond_type'] = df['image_name'].apply(lambda x: 'Circle' if '10191' in x else 'Square')

# Process big prawns data
big_df = pd.DataFrame()
big_df['image_name'] = df['image_name']
big_df['pond_type'] = df['pond_type']
big_df['prawn_type'] = 'Big'
big_df['x'] = df['big_eye_x']
big_df['y'] = df['big_eye_y']
big_df['length'] = df['big_total_length']
big_df['expected'] = expected_big_total
big_df['abs_error'] = abs(big_df['length'] - big_df['expected'])
big_df['error_pct'] = (big_df['abs_error'] / big_df['expected']) * 100

# Process small prawns data
small_df = pd.DataFrame()
small_df['image_name'] = df['image_name']
small_df['pond_type'] = df['pond_type']
small_df['prawn_type'] = 'Small'
small_df['x'] = df['small_eye_x']
small_df['y'] = df['small_eye_y']
small_df['length'] = df['small_total_length']
small_df['expected'] = expected_small_total
small_df['abs_error'] = abs(small_df['length'] - small_df['expected'])
small_df['error_pct'] = (small_df['abs_error'] / small_df['expected']) * 100

# Drop rows with NaN values in key columns
big_df = big_df.dropna(subset=['x', 'y', 'length'])
small_df = small_df.dropna(subset=['x', 'y', 'length'])

# Simple outlier removal for coordinates (Z-score > 3)
def remove_outliers(df, columns=['x', 'y']):
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[abs(df[col] - mean) <= 3 * std]
    return df

# Apply outlier removal only if you need it
# big_df = remove_outliers(big_df)
# small_df = remove_outliers(small_df)

# Combine the dataframes
plot_df = pd.concat([big_df, small_df], ignore_index=True)

# Compute some statistics for debugging
print(f"Total data points: {len(plot_df)}")
print(f"Circle pond points: {len(plot_df[plot_df['pond_type'] == 'Circle'])}")
print(f"Square pond points: {len(plot_df[plot_df['pond_type'] == 'Square'])}")
print(f"Big prawn points: {len(plot_df[plot_df['prawn_type'] == 'Big'])}")
print(f"Small prawn points: {len(plot_df[plot_df['prawn_type'] == 'Small'])}")

# Max error for color scale
max_error = 20  # Adjust based on your data

# Create figure with facet_col for pond type
fig = px.scatter(
    plot_df,
    x='x',
    y='y',
    color='error_pct',
    symbol='prawn_type',
    facet_col='pond_type',
    color_continuous_scale='RdYlGn_r',  # Red for high error, green for low
    range_color=[0, max_error],
    hover_name='image_name',
    hover_data={
        'length': ':.1f',
        'expected': ':.1f',
        'error_pct': ':.1f',
        'prawn_type': True,
        'x': False,
        'y': False,
        'pond_type': False
    },
    labels={
        'error_pct': 'Error %',
        'x': 'Eye X Position',
        'y': 'Eye Y Position'
    },
    title='Total Length Error by Spatial Location',
    symbol_map={'Big': 'triangle-up', 'Small': 'circle'}
)

# Update marker size - same for all points
fig.update_traces(marker=dict(size=12))

# Update layout
fig.update_layout(
    height=600,
    width=1200,
    margin=dict(l=50, r=150, t=80, b=50),  # Extra right margin for color bar
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    coloraxis_colorbar=dict(
        title='Error %',
        x=1.0,  # Position at far right
        y=0.5,  # Center vertically
        len=0.9,  # Long color bar as requested
        thickness=20  # Thicker bar for visibility
    )
)

# Clean up facet labels
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

#save the plot
fig.write_html("prawn_error_analysis.html")

# Save figure to file (optional)
# fig.write_html("prawn_error_analysis.html")
# fig.write_image("prawn_error_analysis.png", scale=2)

# Show figure
fig.show()