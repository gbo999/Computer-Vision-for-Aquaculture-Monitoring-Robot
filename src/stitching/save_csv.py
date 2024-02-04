import pandas as pd
import os

# Get the current working directory
work_dir = os.getcwd()

# Specify the path to your CSV file
csv_file = f'{work_dir}/src/stitching/output/stats.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file,delimiter=';')

# Display the DataFrame
print(df)