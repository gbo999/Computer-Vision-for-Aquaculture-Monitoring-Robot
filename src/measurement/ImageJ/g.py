import pandas as pd
import numpy as np
import os


os.chdir(r"C:\Users\gbo10\Videos\research\counting_research_algorithms\src\measurement\ImageJ")

# Load the processed data from the Excel files
final_data_1 = pd.read_excel("final_full_data_1_with_prawn_ids.xlsx")
final_data_2 = pd.read_excel("final_full_data_2_with_prawn_ids.xlsx")
final_data_3 = pd.read_excel("final_full_data_3_with_prawn_ids.xlsx")

# Create a set of (Label, PrawnID) tuples for each dataset
pairs_1 = set(zip(final_data_1['Label'], final_data_1['PrawnID']))
pairs_2 = set(zip(final_data_2['Label'], final_data_2['PrawnID']))
pairs_3 = set(zip(final_data_3['Label'], final_data_3['PrawnID']))

# Find (Label, PrawnID) pairs that appear in all three datasets
valid_pairs = pairs_1.intersection(pairs_2, pairs_3)

print(f"Number of valid (Label, PrawnID) pairs: {len(valid_pairs)}")

# Filter each dataset to keep only valid (Label, PrawnID) pairs
filtered_data_1 = final_data_1[final_data_1.set_index(['Label', 'PrawnID']).index.isin(valid_pairs)]
filtered_data_2 = final_data_2[final_data_2.set_index(['Label', 'PrawnID']).index.isin(valid_pairs)]
filtered_data_3 = final_data_3[final_data_3.set_index(['Label', 'PrawnID']).index.isin(valid_pairs)]

# Merge the filtered datasets
merged_data = pd.merge(filtered_data_1, filtered_data_2, on=['Label', 'PrawnID'], suffixes=('_1', '_2'))
merged_data = pd.merge(merged_data, filtered_data_3, on=['Label', 'PrawnID'], suffixes=('', '_3'))

# Rename columns from final_data_3 that didn't get a suffix
columns_to_rename = [col for col in merged_data.columns if not col.endswith(('_1', '_2', '_3')) and col not in ['Label', 'PrawnID']]
rename_dict = {col: f"{col}_3" for col in columns_to_rename}
merged_data = merged_data.rename(columns=rename_dict)

print("\nSample of merged_data:")
print(merged_data.head())
def get_scale(data, label, dataset_suffix):
    # Get the second row for the given label and extract the scale
    return data[(data['Label'] == label) & (data.index == data[data['Label'] == label].index[1])]['Length'].values[0] 


def calculate_statistics(row):
    lengths = [row['Length_1'], row['Length_2'], row['Length_3']]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    scale_1 = get_scale(final_data_1, row['Label'], '_1')
    scale_2 = get_scale(final_data_2, row['Label'], '_2')
    scale_3 = get_scale(final_data_3,row['Label'], '_3')
    
    result = {
        'Label': row['Label'],
        'PrawnID': row['PrawnID'],
        'Avg_Length': avg_length,
        'Std_Length': std_length,
        'Uncertainty': std_length / (3 ** 0.5) if std_length != 0 else 0,
        'Length_1': row['Length_1'],
        'Length_2': row['Length_2'],
        'Length_3': row['Length_3'],
        'BoundingBox_1': (row['BX_1'], row['BY_1'], row['Width_1'], row['Height_1']),
        'BoundingBox_2': (row['BX_2'], row['BY_2'], row['Width_2'], row['Height_2']),
        'BoundingBox_3': (row['BX_3'], row['BY_3'], row['Width_3'], row['Height_3']),
        'Angle_1': row['Angle_1'],
        'Angle_2': row['Angle_2'],
        'Angle_3': row['Angle_3'],
        'Scale_1': scale_1,
        'Scale_2': scale_2,
        'Scale_3': scale_3
    }
    return pd.Series(result)

# Apply the statistics calculation to each row
final_statistics = merged_data.apply(calculate_statistics, axis=1)

print("\nSample of final_statistics:")
print(final_statistics.head())

# Calculate global uncertainty (across all measurements)
# Calculate global uncertainty (across all measurements)
all_lengths = pd.concat([
    final_statistics['Length_1'][(final_statistics['PrawnID'].notna()) & (final_statistics['Length_1'].notna())],
    final_statistics['Length_2'][(final_statistics['PrawnID'].notna()) & (final_statistics['Length_2'].notna())],
    final_statistics['Length_3'][(final_statistics['PrawnID'].notna()) & (final_statistics['Length_3'].notna())]
])

if len(all_lengths) > 0:
    global_std = all_lengths.std()
    global_uncertainty = global_std / (len(all_lengths) ** 0.5)
else:
    print("Warning: No valid length measurements found.")
    global_uncertainty = np.nan

# Add global uncertainty to the dataframe
final_statistics['Global_Uncertainty'] = global_uncertainty

# Save the final statistics to an Excel file
final_statistics.to_excel("final_full_statistics_with_prawn_ids_and_uncertainty.xlsx", index=False)

print("\nData saved to final_statistics_with_prawn_ids_and_uncertainty.xlsx")