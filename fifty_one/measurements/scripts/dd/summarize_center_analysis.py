#!/usr/bin/env python3
"""
Summarize the results of the center match analysis.
This script reads the center analysis files and provides a concise summary
of the results, including recommendations for improving the matching.
"""

import os
import pandas as pd
import numpy as np

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(script_dir, '../../..'))

# Data paths
body_file = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/center_analysis/body_center_analysis.csv'
carapace_file = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/center_analysis/carapace_center_analysis.csv'

def print_separator():
    print("=" * 60)

def analyze_bbox_distances(df, name):
    """Analyze the distances between centers and bounding boxes."""
    print_separator()
    print(f"  {name.upper()} MEASUREMENTS ANALYSIS")
    print_separator()
    print(f"Total matches: {len(df)}")
    
    # Centers inside/outside bounding box
    centers_inside = df.center_in_any_bbox.sum()
    pct_inside = centers_inside / len(df) * 100
    print(f"Centers inside bounding box: {centers_inside} ({pct_inside:.1f}%)")
    
    outside_bbox = df[~df.center_in_any_bbox]
    pct_outside = len(outside_bbox) / len(df) * 100
    print(f"Centers outside bounding box: {len(outside_bbox)} ({pct_outside:.1f}%)")
    
    # Distance statistics
    if len(outside_bbox) > 0:
        median_dist = outside_bbox.distance_to_nearest_bbox.median()
        max_dist = outside_bbox.distance_to_nearest_bbox.max()
        min_dist = outside_bbox.distance_to_nearest_bbox.min()
        
        print(f"Distance to nearest bbox (for centers outside):")
        print(f"  Minimum: {min_dist:.1f} pixels")
        print(f"  Median: {median_dist:.1f} pixels")
        print(f"  Maximum: {max_dist:.1f} pixels")
        
        # Group by match type
        match_types = df.groupby('match_type')['center_in_any_bbox'].agg(['count', 'sum'])
        match_types['pct_inside'] = match_types['sum'] / match_types['count'] * 100
        
        print("\nMatch type analysis:")
        for match_type, row in match_types.iterrows():
            print(f"  {match_type}: {row['count']} matches, {row['sum']} inside ({row['pct_inside']:.1f}%)")
        
        # Top 3 largest distances
        print("\nTop 3 largest distances to nearest bounding box:")
        largest_dists = outside_bbox.sort_values('distance_to_nearest_bbox', ascending=False).head(3)
        for i, (_, row) in enumerate(largest_dists.iterrows()):
            print(f"  {i+1}. {row['meas_image_name']}")
            print(f"     Distance: {row['distance_to_nearest_bbox']:.1f} pixels")
            print(f"     Match type: {row['match_type']}")
            print(f"     Nearest bbox: {row['nearest_bbox_column']}")
    
    return {
        'name': name,
        'total': len(df),
        'inside': centers_inside,
        'pct_inside': pct_inside,
        'outside': len(outside_bbox),
        'pct_outside': pct_outside,
        'median_dist': outside_bbox.distance_to_nearest_bbox.median() if len(outside_bbox) > 0 else 0,
        'max_dist': outside_bbox.distance_to_nearest_bbox.max() if len(outside_bbox) > 0 else 0
    }

def main():
    print("\nCENTER MATCH ANALYSIS SUMMARY")
    print("This summary analyzes how well the centers match with bounding boxes.")
    
    try:
        body_df = pd.read_csv(body_file)
        carapace_df = pd.read_csv(carapace_file)
    except Exception as e:
        print(f"Error reading analysis files: {e}")
        return
    
    # Analyze both datasets
    body_stats = analyze_bbox_distances(body_df, "Body")
    carapace_stats = analyze_bbox_distances(carapace_df, "Carapace")
    
    # Overall findings
    print_separator()
    print("  FINDINGS AND RECOMMENDATIONS")
    print_separator()
    
    # Key findings
    print("Key findings:")
    print(f"1. For body measurements, {body_stats['pct_inside']:.1f}% of centers are inside the bounding box.")
    print(f"2. For carapace measurements, only {carapace_stats['pct_inside']:.1f}% of centers are inside the bounding box.")
    
    if body_stats['outside'] > 0:
        print(f"3. The median distance to nearest bbox for body centers outside boxes is {body_stats['median_dist']:.1f} pixels.")
    
    if carapace_stats['outside'] > 0:
        print(f"4. The median distance to nearest bbox for carapace centers outside boxes is {carapace_stats['median_dist']:.1f} pixels.")
    
    print(f"5. The maximum distance between a center and nearest bbox can be very large (up to {max(body_stats['max_dist'], carapace_stats['max_dist']):.1f} pixels).")
    
    # Recommendations
    print("\nRecommendations:")
    print("1. Adjust the tolerance value in combine_lengths_improved.py to improve matching.")
    
    if body_stats['pct_inside'] < 75 or carapace_stats['pct_inside'] < 75:
        print("2. Examine the center extraction logic to ensure centers are correctly positioned relative to the bounding boxes.")
    
    print("3. Check the problem matches files to identify systematic issues:")
    print(f"   - {os.path.basename(body_file.replace('_center_analysis.csv', '_problem_matches.csv'))}")
    print(f"   - {os.path.basename(carapace_file.replace('_center_analysis.csv', '_problem_matches.csv'))}")
    
    print("4. Try running the matching with different tolerance values:")
    print("   python combine_lengths_improved.py --tolerance 600")

if __name__ == "__main__":
    main() 