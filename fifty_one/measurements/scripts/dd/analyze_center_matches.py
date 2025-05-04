import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../../..'))

# Data paths
combined_body_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/combined_body_length_data.csv'
combined_carapace_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/combined_carapace_length_data.csv'

# Output directory for analysis results
output_dir = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/center_analysis'
os.makedirs(output_dir, exist_ok=True)

def is_center_within_bbox(center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h, tolerance=0):
    """Check if a point is inside a bounding box with optional tolerance."""
    if None in (center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h):
        return False
    
    within_x = (bbox_x - tolerance) <= center_x <= (bbox_x + bbox_w + tolerance)
    within_y = (bbox_y - tolerance) <= center_y <= (bbox_y + bbox_h + tolerance)
    
    return within_x and within_y

def calculate_distance_to_bbox(center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h):
    """Calculate the distance from a center point to the nearest point on a bounding box."""
    # Check if center is inside the bbox
    if is_center_within_bbox(center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h):
        return 0.0
    
    # Calculate distances to each edge of the bbox
    dist_left = max(0, bbox_x - center_x)
    dist_right = max(0, center_x - (bbox_x + bbox_w))
    dist_top = max(0, bbox_y - center_y)
    dist_bottom = max(0, center_y - (bbox_y + bbox_h))
    
    # Return the Euclidean distance to the nearest point on the bbox
    if dist_left > 0 and dist_top > 0:  # Top-left corner
        return np.sqrt(dist_left**2 + dist_top**2)
    elif dist_right > 0 and dist_top > 0:  # Top-right corner
        return np.sqrt(dist_right**2 + dist_top**2)
    elif dist_left > 0 and dist_bottom > 0:  # Bottom-left corner
        return np.sqrt(dist_left**2 + dist_bottom**2)
    elif dist_right > 0 and dist_bottom > 0:  # Bottom-right corner
        return np.sqrt(dist_right**2 + dist_bottom**2)
    else:  # Center is outside the bbox but aligned with one of its edges
        return max(dist_left, dist_right, dist_top, dist_bottom)

def analyze_center_matches(combined_data_path, measurement_type='body'):
    """
    Analyze the matches between center points and bounding boxes to identify potential issues.
    
    Args:
        combined_data_path: Path to the combined data CSV
        measurement_type: 'body' or 'carapace'
    """
    print(f"\n--- Analyzing {measurement_type} center matches ---")
    
    # Load combined data
    try:
        df = pd.read_csv(combined_data_path)
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Find bounding box columns and center coordinate columns
    bbox_cols = []
    for col in df.columns:
        if col.startswith('length_BoundingBox') or col.startswith('length_boundingbox'):
            bbox_cols.append(col)
    
    if not bbox_cols:
        print("No bounding box columns found in the dataframe")
        return
    
    print(f"Found {len(bbox_cols)} bounding box columns: {bbox_cols}")
    
    # Check for center coordinate columns
    if 'meas_center_x' in df.columns and 'meas_center_y' in df.columns:
        center_x_col = 'meas_center_x'
        center_y_col = 'meas_center_y'
    elif 'center_x' in df.columns and 'center_y' in df.columns:
        center_x_col = 'center_x'
        center_y_col = 'center_y'
    else:
        print("Could not find center coordinate columns")
        return
    
    # Create columns for analysis results
    df['center_in_any_bbox'] = False
    df['distance_to_nearest_bbox'] = np.nan
    df['nearest_bbox_column'] = None
    
    # Analyze each row
    for idx, row in df.iterrows():
        # Get center coordinates and scale to image dimensions
        center_x = row[center_x_col] * 5312  # Scaling factors from combine_lengths_improved.py
        center_y = row[center_y_col] * 2988
        
        # Analyze each bounding box
        min_distance = float('inf')
        nearest_bbox = None
        
        for bbox_col in bbox_cols:
            if pd.isna(row[bbox_col]):
                continue
            
            bbox_str = str(row[bbox_col])
            if '(' in bbox_str and ')' in bbox_str:
                try:
                    # Extract values from format like (x, y, w, h)
                    bbox_str = bbox_str.strip('()').replace(' ', '')
                    bbox_values = [float(x.strip()) for x in bbox_str.split(',')]
                    if len(bbox_values) == 4:
                        bbox_x, bbox_y, bbox_w, bbox_h = bbox_values
                        
                        # Check if center is in this bbox
                        if is_center_within_bbox(center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h):
                            df.at[idx, 'center_in_any_bbox'] = True
                            df.at[idx, 'distance_to_nearest_bbox'] = 0.0
                            df.at[idx, 'nearest_bbox_column'] = bbox_col
                            min_distance = 0.0
                            nearest_bbox = bbox_col
                            break  # Found a containing bbox, no need to check others
                        
                        # Calculate distance to this bbox
                        distance = calculate_distance_to_bbox(center_x, center_y, bbox_x, bbox_y, bbox_w, bbox_h)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_bbox = bbox_col
                except Exception as e:
                    print(f"Error parsing bounding box {bbox_str} in column {bbox_col} for row {idx}: {e}")
        
        # If we didn't find a containing bbox, record the nearest one
        if not df.at[idx, 'center_in_any_bbox'] and nearest_bbox is not None:
            df.at[idx, 'distance_to_nearest_bbox'] = min_distance
            df.at[idx, 'nearest_bbox_column'] = nearest_bbox
    
    # Summary statistics
    total_rows = len(df)
    centers_in_bbox = df['center_in_any_bbox'].sum()
    centers_outside_bbox = total_rows - centers_in_bbox
    
    print(f"Total rows: {total_rows}")
    print(f"Centers inside a bounding box: {centers_in_bbox} ({centers_in_bbox/total_rows:.1%})")
    print(f"Centers outside all bounding boxes: {centers_outside_bbox} ({centers_outside_bbox/total_rows:.1%})")
    
    # Analyze match types
    match_type_counts = df['match_type'].value_counts()
    print("\nMatch type distribution:")
    for match_type, count in match_type_counts.items():
        print(f"  {match_type}: {count} ({count/total_rows:.1%})")
    
    # Analyze centers inside/outside bbox by match type
    print("\nCenters inside/outside bbox by match type:")
    match_type_bbox_stats = df.groupby('match_type')['center_in_any_bbox'].agg(['count', 'sum'])
    match_type_bbox_stats['pct_inside'] = match_type_bbox_stats['sum'] / match_type_bbox_stats['count']
    print(match_type_bbox_stats)
    
    # Analyze distance distribution for centers outside bounding boxes
    outside_centers = df[~df['center_in_any_bbox']]
    if len(outside_centers) > 0:
        distance_stats = outside_centers['distance_to_nearest_bbox'].describe()
        print("\nDistance statistics for centers outside bounding boxes:")
        print(distance_stats)
        
        # Create a histogram of distances
        plt.figure(figsize=(10, 6))
        plt.hist(outside_centers['distance_to_nearest_bbox'].dropna(), bins=30)
        plt.title(f"{measurement_type} - Distance from Center to Nearest Bounding Box")
        plt.xlabel("Distance (pixels)")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        # Save the histogram
        hist_path = os.path.join(output_dir, f"{measurement_type}_distance_histogram.png")
        plt.savefig(hist_path)
        plt.close()
        print(f"Saved distance histogram to {hist_path}")
    
    # Save analysis results
    output_path = os.path.join(output_dir, f"{measurement_type}_center_analysis.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved analysis results to {output_path}")
    
    # Create a list of potentially problematic matches (centers far from any bbox)
    if len(outside_centers) > 0:
        # Find rows where distance is > 75th percentile
        distance_threshold = distance_stats['75%']
        problem_matches = outside_centers[outside_centers['distance_to_nearest_bbox'] > distance_threshold]
        
        problem_path = os.path.join(output_dir, f"{measurement_type}_problem_matches.csv")
        problem_matches.to_csv(problem_path, index=False)
        print(f"Saved {len(problem_matches)} potential problem matches to {problem_path}")
    
    return df

def main():
    print("Starting center match analysis...")
    
    # Process body data
    if os.path.exists(combined_body_path):
        body_analysis = analyze_center_matches(combined_body_path, 'body')
    else:
        print(f"Body data file not found: {combined_body_path}")
    
    # Process carapace data
    if os.path.exists(combined_carapace_path):
        carapace_analysis = analyze_center_matches(combined_carapace_path, 'carapace')
    else:
        print(f"Carapace data file not found: {combined_carapace_path}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 