import cv2
import os

# Folder containing JPEG images 
jpg_dir = "calib/"

# Output folder for PNG images
png_dir = "png/" 

# List all JPEG files
jpgs = [f for f in os.listdir(jpg_dir) if f.endswith('.jpeg')]

for jpg in jpgs:
    # Load JPEG file
    img = cv2.imread(jpg_dir + jpg)
    
    # Construct PNG filename  
    png = jpg.split('.')[0] + '.png'
    
    # Convert to PNG and save output
    is_success, buffer = cv2.imencode(".png", img)
    with open(png_dir+png, 'wb') as file: 
         file.write(buffer)

print('Done')