import cv2
import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt

# Load the distorted image
image_path = r"C:\Users\gbo10\OneDrive\measurement_paper_images\images to annotatin detection with x any labeling\car images\closed\valid\carapace\dark\gamma\check\GX010162_63_737.jpg_gamma.jpg_gamma.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Could not open or find the distorted image!")
    exit()

# Load the STMap (EXR file)
stmap_path = r"C:\Users\gbo10\OneDrive\measurement_paper_images\images to annotatin detection with x any labeling\car images\closed\valid\carapace\dark\gamma\GX010162_63_737.jpg_gamma.jpg_gamma.jpg-GoPro-HERO11-Black-Wide-undistort-0.exr"
exr_file = OpenEXR.InputFile(stmap_path)

# Extract the metadata and image dimensions
header = exr_file.header()
data_window = header['dataWindow']
dw = Imath.Box2i(data_window.min, data_window.max)
stmap_width, stmap_height = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

# Read the RGB channels
pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
red_channel = exr_file.channel('R', pixel_type)
green_channel = exr_file.channel('G', pixel_type)

# Convert the channels to numpy arrays
map_x = np.frombuffer(red_channel, dtype=np.float32).reshape(stmap_height, stmap_width)
map_y = np.frombuffer(green_channel, dtype=np.float32).reshape(stmap_height, stmap_width)

print(f"STMap X: min={map_x.min()}, max={map_x.max()}")
print(f"STMap Y: min={map_y.min()}, max={map_y.max()}")

# Get the dimensions of the original image
rows, cols = image.shape[:2]

# Scale the STMap values to image coordinates
map_x_scaled = map_x * (cols - 1)
map_y_scaled = map_y * (rows - 1)

# Ensure the coordinates are within the image bounds
map_x_final = np.clip(map_x_scaled, 0, cols - 1).astype(np.float32)
map_y_final = np.clip(map_y_scaled, 0, rows - 1).astype(np.float32)

# Apply the undistortion
undistorted_image = cv2.remap(image, map_x_final, map_y_final, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# Save the undistorted image
output_path = r"C:\Users\gbo10\OneDrive\measurement_paper_images\images to annotatin detection with x any labeling\car images\closed\valid\carapace\dark\gamma\check\GX010162_63_737_undistorted.jpg"
cv2.imwrite(output_path, undistorted_image)

# Visualize the results


# Print debug information
print("Final map_x min:", np.min(map_x_final), "max:", np.max(map_x_final))
print("Final map_y min:", np.min(map_y_final), "max:", np.max(map_y_final))
print("Displaced X at (0,0):", map_x_final[0, 0])
print("Displaced Y at (0,0):", map_y_final[0, 0])