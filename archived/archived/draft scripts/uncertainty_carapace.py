import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create directory for figures if it doesn't exist
os.makedirs('fifty_one/figures', exist_ok=True)

# Read the CSV file
df = pd.read_csv('fifty_one/re (2)/Carapace20times.csv')

# Calculate mean and standard deviation for Length measurement
stats = {'mean': df['Length'].mean(), 'std': df['Length'].std()}

# Create figure for Length measurement
plt.figure(figsize=(8, 6))
plt.suptitle('Carapace Length Measurement Uncertainty Analysis', fontsize=16)

# Create distribution plot
sns.histplot(data=df, x='Length', kde=True)

# Add vertical lines for mean and std ranges
plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.2f}')
plt.axvline(stats['mean'] - stats['std'], color='green', linestyle=':', 
           label=f'±1σ: {stats["std"]:.2f}')
plt.axvline(stats['mean'] + stats['std'], color='green', linestyle=':')

# Calculate coefficient of variation (CV)
cv = (stats['std'] / stats['mean']) * 100

plt.title(f'Carapace Length Distribution (CV: {cv:.2f}%)')
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('fifty_one/figures/carapace_uncertainty.png')
plt.close()

# Print summary statistics
print("\nCarapace Length Measurement Uncertainty Summary:")
print("-" * 50)
print(f"Mean: {stats['mean']:.2f}")
print(f"Standard Deviation: {stats['std']:.2f}") 
print(f"Coefficient of Variation: {cv:.2f}%") 