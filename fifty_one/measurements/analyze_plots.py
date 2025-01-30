import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import median_abs_deviation
import numpy as np

def create_violin_plots():
    # Read the data
    circle2_df = pd.read_csv("fifty_one/measurements/measurement_analysis_circle2.csv")
    square_df = pd.read_csv("fifty_one/measurements/measurement_analysis_square.csv")
    
    # Combine dataframes
    circle2_df['position'] = 'circle2'
    square_df['position'] = 'square'
    df_combined = pd.concat([circle2_df, square_df])
    
    # Function to calculate MAD
    def calc_mad(x):
        return median_abs_deviation(x.dropna(), scale='normal')
    
    # Calculate statistics for all groups
    stats = df_combined.groupby(['position', 'which']).agg({
        'total_diff_mm': [
            ('count', 'count'),
            ('median', 'median'),
            ('mad', calc_mad),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75))
        ]
    })
    
    # Print statistics
    for idx in stats.index:
        position, size = idx
        group = f"{size} {position}"
        row = stats.loc[idx]
        print(f"\n{group} statistics:")
        print(f"Count: {row[('total_diff_mm', 'count')]}")
        print(f"Median: {row[('total_diff_mm', 'median')]:.2f} ± {row[('total_diff_mm', 'mad')]:.2f} mm (MAD)")
        print(f"IQR: {row[('total_diff_mm', 'q3')] - row[('total_diff_mm', 'q1')]:.2f} mm")
        print(f"(Q1: {row[('total_diff_mm', 'q1')]:.2f}, Q3: {row[('total_diff_mm', 'q3')]:.2f})")
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Total Length Differences', 'Carapace Length Differences'))

    # Add traces for total length
    for group in ['big circle2', 'small circle2', 'big square', 'small square']:
        position = group.split()[1]
        size = group.split()[0]
        
        data = df_combined[
            (df_combined['position'] == position) & 
            (df_combined['which'] == size)
        ]['total_diff_mm']
        
        fig.add_trace(
            go.Violin(
                y=data,
                name=group,
                box_visible=True,
                meanline_visible=True,
                points='all',  # Show all points
                jitter=0.05,   # Add jitter to points
                showlegend=False
            ),
            row=1, col=1
        )

    # Add traces for carapace length
    for group in ['big circle2', 'small circle2', 'big square', 'small square']:
        position = group.split()[1]
        size = group.split()[0]
        
        data = df_combined[
            (df_combined['position'] == position) & 
            (df_combined['which'] == size)
        ]['carapace_diff_mm']
        
        fig.add_trace(
            go.Violin(
                y=data,
                name=group,
                box_visible=True,
                meanline_visible=True,
                points='all',  # Show all points
                jitter=0.05,   # Add jitter to points
                showlegend=True
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title='Distribution of Measurement Differences by Group',
        height=1000,
        showlegend=True,
        violinmode='group',
        template='plotly_white'  # Clean white background
    )
    
    # Add zero reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_yaxes(title_text='Difference (mm)', row=1, col=1)
    fig.update_yaxes(title_text='Difference (mm)', row=2, col=1)

    # Show plot
    fig.show()

# Run the analysis
create_violin_plots() 