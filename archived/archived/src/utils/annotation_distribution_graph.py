import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# Suppress specific NumPy warnings that don't affect our plot
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       module="numpy.core")

# Set the style
plt.style.use('ggplot')
sns.set(font_scale=1.8)  # Increase font scale even more
plt.rcParams['figure.figsize'] = (14, 10)  # Larger figure size

# Data from the graph
categories = ['0', '1', '2-6', '7-11', '12-16', '17-21']
frequencies = [81, 70, 356, 144, 22, 2]
total_images = 675  # Total number of images

# Create the plot
fig, ax = plt.subplots()

# Plot with narrower bars but keep edgecolor
bars = ax.bar(categories, frequencies, width=0.4, edgecolor='black', 
              linewidth=1.5, color='lightgray')

# Add the frequency values INSIDE each bar
for bar in bars:
    height = bar.get_height()
    # Only display the number if there's enough space in the bar
    if height > 10:  # Minimum height check to fit text
        # Position text in the middle of the bar
        ax.text(bar.get_x() + bar.get_width()/2., 
                height/2,  # Middle of the bar
                f'{int(height)}',  # Format as integer
                ha='center', va='center', fontsize=18, fontweight='bold', 
                color='black')  # Black text to contrast with light gray bars

# Add labels and title with larger font
ax.set_xlabel('Annotation Count', fontsize=24, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=24, fontweight='bold')
ax.set_title(f'Distribution of Annotation Counts\nTotal Images: {total_images}', 
             fontsize=28, fontweight='bold')

# Set y-axis limit to match the original graph
ax.set_ylim(0, 370)

# Add grid lines for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)  # Place gridlines behind bars

# Increase tick label size
ax.tick_params(axis='both', which='major', labelsize=22)

# Add a light border around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('gray')
    spine.set_linewidth(0.5)

# Adjust layout to make sure everything fits
plt.tight_layout()

# Save the figure with high resolution
plt.savefig('annotation_distribution.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show() 