import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
df = pd.read_csv('data.csv')

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='vertices', y='edges', alpha=0.6)

# Add trend line
z = np.polyfit(df['vertices'], df['edges'], 1)
p = np.poly1d(z)
plt.plot(df['vertices'], p(df['vertices']), "r--", alpha=0.8)

# Calculate correlation coefficient
correlation = df['vertices'].corr(df['edges'])

# Add labels and title
plt.xlabel('Number of Vertices')
plt.ylabel('Number of Edges')
plt.title(f'Relationship between Vertices and Edges\nCorrelation: {correlation:.3f}')

# Save the plot
plt.savefig('relationship.png')
plt.close()

# Print statistical summary
print("\nStatistical Summary:")
print(df.describe())
print(f"\nCorrelation coefficient: {correlation:.3f}")

# Calculate the average ratio of edges to vertices
df['ratio'] = df['edges'] / df['vertices']
print(f"\nAverage edges per vertex: {df['ratio'].mean():.3f}") 