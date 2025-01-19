import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import median_abs_deviation

def create_violin_plots():
    # Read the data
    right_df = pd.read_csv("fifty_one/measurements/measurement_analysis_right.csv")
    square_df = pd.read_csv("fifty_one/measurements/measurement_analysis_square.csv")
    
    # Print data info
    print("\nRight DataFrame Info:")
    print(right_df[['which', 'total_diff_mm', 'carapace_diff_mm']].describe())
    
    print("\nSquare DataFrame Info:")
    print(square_df[['which', 'total_diff_mm', 'carapace_diff_mm']].describe())
    
    # Combine dataframes
    right_df['position'] = 'right'
    square_df['position'] = 'square'
    df_combined = pd.concat([right_df, square_df])
    
    # Print value ranges for each group
    for group in ['big right', 'small right', 'big square', 'small square']:
        position = group.split()[1]
        size = group.split()[0]
        
        subset = df_combined[
            (df_combined['position'] == position) & 
            (df_combined['which'] == size)
        ]
        
        # Calculate IQR
        Q1 = subset['total_diff_mm'].quantile(0.25)
        Q3 = subset['total_diff_mm'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate MAD
        mad = median_abs_deviation(subset['total_diff_mm'].dropna(), scale='normal')
        
        print(f"\n{group} statistics:")
        print(f"Median: {subset['total_diff_mm'].median():.2f} ± {mad:.2f} (MAD)")
        print(f"IQR: {IQR:.2f} (Q1: {Q1:.2f}, Q3: {Q3:.2f})")
        print(f"Range: {subset['total_diff_mm'].min():.2f} to {subset['total_diff_mm'].max():.2f}")
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=('Total Length Differences', 'Carapace Length Differences'))

    # Add traces for total length
    for group in ['big right', 'small right', 'big square', 'small square']:
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
                showlegend=False
            ),
            row=1, col=1
        )

    # Add traces for carapace length
    for group in ['big right', 'small right', 'big square', 'small square']:
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
                showlegend=True
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title='Distribution of Measurement Differences by Group',
        height=1000,
        showlegend=True,
        violinmode='group'
    )
    
    fig.update_yaxes(title_text='Difference (mm)', row=1, col=1)
    fig.update_yaxes(title_text='Difference (mm)', row=2, col=1)

    # Show plot
    fig.show()

# Run the analysis
create_violin_plots() 