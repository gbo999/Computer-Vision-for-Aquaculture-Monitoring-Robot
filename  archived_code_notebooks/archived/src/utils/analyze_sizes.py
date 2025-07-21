import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory containing the prediction files
labels_dir = r'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/runs/pose/predict51/labels'

# Lists to store dimensions
widths = []
heights = []
aspect_ratios = []
areas = []

# Process all prediction files
for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            # Extract dimensions
            width = float(parts[3])
            height = float(parts[4])
            
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(height/width)
            areas.append(width * height)

# Convert to numpy arrays
widths = np.array(widths)
heights = np.array(heights)
aspect_ratios = np.array(aspect_ratios)
areas = np.array(areas)

# Print statistics
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

print("\nAspect Ratio statistics (height/width):")
print(f"Min: {aspect_ratios.min():.4f}")
print(f"Max: {aspect_ratios.max():.4f}")
print(f"Mean: {aspect_ratios.mean():.4f}")
print(f"Median: {np.median(aspect_ratios):.4f}")

# Create visualizations
plt.figure(figsize=(15, 10))

# Width histogram
plt.subplot(2, 2, 1)
plt.hist(widths, bins=50)
plt.title('Width Distribution')
plt.xlabel('Width (relative)')
plt.ylabel('Count')

# Height histogram
plt.subplot(2, 2, 2)
plt.hist(heights, bins=50)
plt.title('Height Distribution')
plt.xlabel('Height (relative)')
plt.ylabel('Count')

# Scatter plot
plt.subplot(2, 2, 3)
plt.scatter(widths, heights, alpha=0.5)
plt.title('Width vs Height')
plt.xlabel('Width')
plt.ylabel('Height')

# Aspect ratio histogram
plt.subplot(2, 2, 4)
plt.hist(aspect_ratios, bins=50)
plt.title('Aspect Ratio Distribution')
plt.xlabel('Aspect Ratio (height/width)')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('size_analysis.png')
print("\nVisualization saved as 'size_analysis.png'") 