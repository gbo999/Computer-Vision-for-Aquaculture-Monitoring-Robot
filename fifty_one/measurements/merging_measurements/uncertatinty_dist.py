import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('fifty_one/shai_data/Square20times.csv')


# Calculate mean and standard deviation for each measurement type
measurement_stats = {
    'Length': {'mean': df['Length'].mean(), 'std': df['Length'].std()}
}

# Create figure for Length measurement
plt.figure(figsize=(8, 6))
plt.suptitle('Length Measurement Uncertainty Analysis', fontsize=16)

# Create distribution plot
sns.histplot(data=df, x='Length', kde=True)

# Get statistics for Length
stats = measurement_stats['Length']
cv = (stats['std'] / stats['mean']) * 100
# Add vertical lines for mean and std ranges
plt.axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats["mean"]:.2f}')
plt.axvline(stats['mean'] - stats['std'], color='green', linestyle=':', 
           label=f'±1σ: {stats["std"]:.2f}')
plt.axvline(stats['mean'] + stats['std'], color='green', linestyle=':')



# Calculate coefficient of variation (CV)




plt.title('Length Distribution')
plt.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nLength Measurement Uncertainty Summary:")
print("-" * 50)
cv = (stats['std'] / stats['mean']) * 100
print(f"\nLength:")
print(f"Mean: {stats['mean']:.2f}")
print(f"Standard Deviation: {stats['std']:.2f}") 
print(f"Coefficient of Variation: {cv:.2f}%")

#calculate uncertainty






