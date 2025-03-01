import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory containing the prediction files
labels_dir = r'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/runs/pose/predict51/labels'

# Dictionary to store dimensions for each video group
groups = defaultdict(lambda: {'widths': [], 'heights': [], 'aspect_ratios': [], 'areas': []})

# Process all prediction files
for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
    # Extract video group (e.g., GX010191, GX010192, etc.)
    basename = os.path.basename(label_file)
    video_group = basename.split('_')[1]  # Get GX010191, etc.
    
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            # Extract dimensions
            width = float(parts[3])
            height = float(parts[4])
            
            # Store dimensions in corresponding group
            groups[video_group]['widths'].append(width)
            groups[video_group]['heights'].append(height)
            groups[video_group]['aspect_ratios'].append(height/width)
            groups[video_group]['areas'].append(width * height)

# Create a figure for each video group
for video_group in sorted(groups.keys()):
    # Convert lists to numpy arrays
    widths = np.array(groups[video_group]['widths'])
    heights = np.array(groups[video_group]['heights'])
    aspect_ratios = np.array(groups[video_group]['aspect_ratios'])
    areas = np.array(groups[video_group]['areas'])
    
    print(f"\n=== Statistics for {video_group} ===")
    print(f"Number of detections: {len(widths)}")
    
    print("\nWidth statistics:")
    print(f"Min: {widths.min():.4f}")
    print(f"Max: {widths.max():.4f}")
    print(f"Mean: {widths.mean():.4f}")
    print(f"Median: {np.median(widths):.4f}")
    print(f"25th percentile: {np.percentile(widths, 25):.4f}")
    print(f"75th percentile: {np.percentile(widths, 75):.4f}")

    print("\nHeight statistics:")
    print(f"Min: {heights.min():.4f}")
    print(f"Max: {heights.max():.4f}")
    print(f"Mean: {heights.mean():.4f}")
    print(f"Median: {np.median(heights):.4f}")
    print(f"25th percentile: {np.percentile(heights, 25):.4f}")
    print(f"75th percentile: {np.percentile(heights, 75):.4f}")
    
    # Create visualizations for this group
    plt.figure(figsize=(15, 10))
    plt.suptitle(f'Size Analysis for {video_group}')
    
    # Width histogram
    plt.subplot(2, 2, 1)
    plt.hist(widths, bins=30)
    plt.title('Width Distribution')
    plt.xlabel('Width (relative)')
    plt.ylabel('Count')

    # Height histogram
    plt.subplot(2, 2, 2)
    plt.hist(heights, bins=30)
    plt.title('Height Distribution')
    plt.xlabel('Height (relative)')
    plt.ylabel('Count')

    # Scatter plot
    plt.subplot(2, 2, 3)
    plt.scatter(widths, heights, alpha=0.5)
    plt.title('Width vs Height')
    plt.xlabel('Width')
    plt.ylabel('Height')

    # Add example sizes
    sizes = [(0.05, 0.15), (0.1, 0.3), (0.15, 0.45)]  # Example size pairs (width, height)
    for w, h in sizes:
        plt.plot([w], [h], 'r*', markersize=10)
        plt.annotate(f'({w:.2f}, {h:.2f})', (w, h), xytext=(10, 10), 
                    textcoords='offset points')

    # Area histogram
    plt.subplot(2, 2, 4)
    plt.hist(areas, bins=30)
    plt.title('Area Distribution')
    plt.xlabel('Area (relative)')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(f'size_analysis_{video_group}.png')

print("\nVisualization saved as separate PNG files for each video group") 