import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel('/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/processed_data/combined_carapace_length_data.xlsx')

# Remove duplicates based on image name
df_no_duplicates = df.drop_duplicates(subset=['meas_image_name'])
df = df_no_duplicates

# Calculate median and MAD
cols = ['meas_scaled_meas_1', 'meas_scaled_meas_2', 'meas_scaled_meas_3', 
        'length_Length_1', 'length_Length_2', 'length_Length_3']
df['median_all'] = df[cols].median(axis=1)
df['MAD_all'] = df[cols].apply(lambda row: (row - row.median()).abs().median(), axis=1)

# Calculate errors
df['error'] = (df['length_Length_fov(mm)'] - df['median_all']).abs()
df['error_percent'] = (df['error'] / df['median_all']) * 100

# Identify outliers (error > 2 * MAD)
df['is_outlier'] = df['error'] > 2 * df['MAD_all']
df['error_to_MAD_ratio'] = df['error'] / df['MAD_all']

# Calculate quality metrics
print(f"Total measurements: {len(df)}")
print(f"Mean error: {df['error'].mean():.2f} mm")
print(f"Median error: {df['error'].median():.2f} mm")
print(f"Mean percentage error: {df['error_percent'].mean():.2f}%")
print(f"Mean MAD: {df['MAD_all'].mean():.2f} mm")

# Print match type distribution
print("\nMatch type distribution:")
match_counts = df['match_type'].value_counts()
for match_type, count in match_counts.items():
    print(f"  {match_type}: {count} ({count/len(df)*100:.1f}%)")

# Analyze error by match type
print("\nError by match type:")
for match_type in df['match_type'].unique():
    subset = df[df['match_type'] == match_type]
    print(f"  {match_type}: Mean error = {subset['error'].mean():.2f} mm, Mean % error = {subset['error_percent'].mean():.2f}%")

# Identify potential problematic matches
outliers = df[df['is_outlier']]
print(f"\nFound {len(outliers)} potential problematic matches ({len(outliers)/len(df)*100:.1f}%)")

# Display the worst offenders
print("\nWorst 10 matches by error-to-MAD ratio:")
worst_matches = df.sort_values('error_to_MAD_ratio', ascending=False).head(10)
print(worst_matches[['meas_image_name', 'match_type', 'median_all', 'length_Length_fov(mm)', 'error', 'MAD_all', 'error_to_MAD_ratio']])

# Visualize the error distribution
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.hist(df['error'], bins=20)
plt.title('Absolute Error Distribution (mm)')
plt.xlabel('Absolute Error (mm)')
plt.ylabel('Count')

plt.subplot(2, 2, 2)
plt.hist(df['error_percent'], bins=20)
plt.title('Percentage Error Distribution')
plt.xlabel('Percentage Error (%)')
plt.ylabel('Count')

plt.subplot(2, 2, 3)
plt.scatter(df['median_all'], df['length_Length_fov(mm)'], alpha=0.5)
plt.plot([0, df['median_all'].max()], [0, df['median_all'].max()], 'r--')
plt.title('Model vs Median Measurements')
plt.xlabel('Median Measurement (mm)')
plt.ylabel('Model Measurement (mm)')
plt.axis('equal')

plt.subplot(2, 2, 4)
plt.scatter(df['MAD_all'], df['error'], alpha=0.5)
plt.title('Error vs MAD')
plt.xlabel('MAD (mm)')
plt.ylabel('Absolute Error (mm)')
plt.plot([0, df['MAD_all'].max()], [0, df['MAD_all'].max()], 'r--')

plt.tight_layout()
plt.savefig('error_analysis.png')
plt.show()

# Categorize errors by magnitude relative to MAD
df['error_category'] = pd.cut(
    df['error_to_MAD_ratio'], 
    bins=[0, 1, 2, 5, float('inf')],
    labels=['within MAD', '1-2x MAD', '2-5x MAD', '>5x MAD']
)

# Print error categories
print("\nError categories:")
error_cats = df['error_category'].value_counts()
for cat, count in error_cats.items():
    print(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")

# Export problematic matches for manual review
outliers.to_excel('potential_problematic_matches.xlsx', index=False) 