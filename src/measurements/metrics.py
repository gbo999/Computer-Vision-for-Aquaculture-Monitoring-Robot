import pandas as pd

file_path = 'C:\\Users\\gbo10\\Videos\\research\\counting_research_algorithms\\src\\measurements\\path_to_your_updated_file.csv'
data = pd.read_csv(file_path)

data['by pixel']=data['Pixel Length']*(10/13)
# Calculations
num_images = len(data['Image File Name'].unique())
num_prawns = len(data)

avg_length = data['Length_mm'].mean()
avg_length2 = data['by pixel'].mean()
std_dev2 = data['by pixel'].std()
std_dev = data['Length_mm'].std()
mae = (data['Length_mm'] - data['Conversion Length']).abs().mean()  # Mean Absolute Error
mae2=(data['Length_mm'] - data['by pixel']).abs().mean()
error_percentage2 = (mae2 / data['Length_mm'].mean()) * 100
error_percentage = (mae / data['Length_mm'].mean()) * 100

print(f'mae2: {mae2}')
print(f'error_percentage2: {error_percentage2}')

print(f'Number of images: {num_images}')
print(f'Number of prawns: {num_prawns}')
print(f'Average length: {avg_length}')
print(f'Standard deviation: {std_dev}')
print(f'Mean Absolute Error: {mae}')
print(f'Error percentage: {error_percentage}')
print(f'avg_length2: {avg_length2}')
print(f'std_dev2: {std_dev2}')