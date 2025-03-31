import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Make sure the output directory exists
os.makedirs('runs/pose/predict57', exist_ok=True)

# Load the data
df = pd.read_csv('runs/pose/predict57/length_analysis.csv')

# Add pond type column
df['pond_type'] = df['image_name'].apply(lambda x: 'Circle' if '10191' in x else 'Square')

# Expected values
expected_big_total = 180  # mm
expected_small_total = 145  # mm

# Calculate error percentages
df['big_error_pct'] = abs(df['big_total_length'] - expected_big_total) / expected_big_total * 100
df['small_error_pct'] = abs(df['small_total_length'] - expected_small_total) / expected_small_total * 100

# Function to categorize errors (for color coding)
def categorize_error(error):
    if pd.isna(error):
        return np.nan
    elif error < 5:
        return '<5%'
    elif error < 10:
        return '5-10%'
    else:
        return '>10%'

# Add error categories
df['big_error_category'] = df['big_error_pct'].apply(categorize_error)
df['small_error_category'] = df['small_error_pct'].apply(categorize_error)

# Create figure with subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Circle Pond - Big Prawns", "Square Pond - Big Prawns",
                    "Circle Pond - Small Prawns", "Square Pond - Small Prawns"),
    vertical_spacing=0.15
)

# Colors for different error ranges
colors = {'<5%': '#2ecc71', '5-10%': '#f1c40f', '>10%': '#e74c3c'}

# Configure subplots
subplot_configs = [
    ('Circle', 'big_error_pct', 'big_error_category', 1, 1),
    ('Square', 'big_error_pct', 'big_error_category', 1, 2),
    ('Circle', 'small_error_pct', 'small_error_category', 2, 1),
    ('Square', 'small_error_pct', 'small_error_category', 2, 2)
]

# Process each subplot
for pond_type, error_col, error_cat_col, row, col in subplot_configs:
    # Filter data for this subplot
    mask = (df['pond_type'] == pond_type) & df[error_col].notna()
    plot_df = df[mask].copy()
    
    if len(plot_df) == 0:
        continue
    
    # Group points by category for coloring
    for category in ['<5%', '5-10%', '>10%']:
        cat_df = plot_df[plot_df[error_cat_col] == category]
        if len(cat_df) > 0:
            # Sort by error value for better visualization
            cat_df = cat_df.sort_values(by=error_col)
            
            # Create y-values to spread points vertically (to avoid overlap)
            y_values = np.linspace(0.1, 0.9, len(cat_df))
            
            # Add scatter points
            fig.add_trace(
                go.Scatter(
                    x=cat_df[error_col],
                    y=y_values,
                    mode='markers',
                    marker=dict(
                        color=colors[category],
                        size=10,
                        line=dict(width=1, color='black')
                    ),
                    name=category,
                    legendgroup=category,
                    showlegend=(row == 1 and col == 1),  # Only show in legend once
                    hovertemplate=
                    "Image: %{customdata}<br>" +
                    "Error: %{x:.1f}%<br>" +
                    "<extra></extra>",
                    customdata=cat_df['image_name']
                ),
                row=row, col=col
            )

# Update layout
fig.update_layout(
    height=800,
    width=1200,
    title_text="Error Distribution by Pond Type and Prawn Size",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5,
        title=dict(text="")
    ),
    margin=dict(t=120, b=80)
)

# Update axes
for i in range(1, 3):
    for j in range(1, 3):
        fig.update_xaxes(
            title_text="Error (%)",
            row=i, col=j,
            range=[0, max(df['big_error_pct'].max(), df['small_error_pct'].max()) * 1.1]
        )
        fig.update_yaxes(
            showticklabels=False,  # Hide y-axis labels as they're meaningless
            title_text="",
            row=i, col=j
        )
        
        # Add vertical lines at 5% and 10% boundaries
        fig.add_shape(
            type="line",
            x0=5, x1=5,
            y0=0, y1=1,
            line=dict(color="gray", width=1, dash="dash"),
            row=i, col=j
        )
        fig.add_shape(
            type="line",
            x0=10, x1=10,
            y0=0, y1=1,
            line=dict(color="gray", width=1, dash="dash"),
            row=i, col=j
        )
        
        # Add text labels for the boundaries
        fig.add_annotation(
            x=5, y=0.95,
            text="5%",
            showarrow=False,
            row=i, col=j
        )
        fig.add_annotation(
            x=10, y=0.95,
            text="10%",
            showarrow=False,
            row=i, col=j
        )

# Save and show the figure
fig.write_html("runs/pose/predict57/error_distribution.html")
print(f"Visualization saved to runs/pose/predict57/error_distribution.html")

# Print summary statistics
print("\nError Distribution Summary:")
print("\nBig Prawns:")
for pond_type in ['Circle', 'Square']:
    pond_data = df[df['pond_type'] == pond_type]
    total = len(pond_data[pond_data['big_total_length'].notna()])
    print(f"\n{pond_type} Pond (n={total}):")
    error_counts = pond_data['big_error_category'].value_counts()
    for category in ['<5%', '5-10%', '>10%']:
        count = error_counts.get(category, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{category}: {count} samples ({percentage:.1f}%)")

print("\nSmall Prawns:")
for pond_type in ['Circle', 'Square']:
    pond_data = df[df['pond_type'] == pond_type]
    total = len(pond_data[pond_data['small_total_length'].notna()])
    print(f"\n{pond_type} Pond (n={total}):")
    error_counts = pond_data['small_error_category'].value_counts()
    for category in ['<5%', '5-10%', '>10%']:
        count = error_counts.get(category, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{category}: {count} samples ({percentage:.1f}%)")