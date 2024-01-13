import pandas as pd
import math
import re
# Load the CSV file
from src.measurements import measurements_calculator
# Load the CSV file without header
data = pd.read_csv('C:\\Users\\gbo10\\Videos\\research\\counting_research_algorithms\\src\\measurements\\data.csv', header=None)

# Assign column names
data.columns = ['Image File Name', 'Pixel Length', 'Conversion Length', 'Num of Squares']

# Function to calculate the length in mm
def calculate_length_mm(row):
    num_squares_str = row['Num of Squares']
    num_squares = float(re.sub("[^0-9.]", "", num_squares_str))
    diagonal = 'diag' in num_squares_str
    square_size_mm = 10

    if diagonal:
        return num_squares * square_size_mm * math.sqrt(2)
    else:
        return num_squares * square_size_mm

# Apply the function to each row
data['Length_mm'] = data.apply(calculate_length_mm, axis=1)

data['Conversion Length'] = data['Pixel Length'].apply(measurements_calculator.convert_pixel_to_real_length)


# Save the updated DataFrame to a new CSV file
data.to_csv('path_to_your_updated_file.csv', index=False)
