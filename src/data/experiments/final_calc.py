import numpy as np

# Camera specifications
camera_height_cm = 73.5  # Camera height from the floor in centimeters
horizontal_fov_degrees = 70.5  # Horizontal field of view in degrees
vertical_fov_degrees = 55.9  # Vertical field of view in degrees

# Image resolution
resolution_width, resolution_height = 1200, 1600  # Assuming portrait orientation

# Calculate the physical FOV widths at the floor level
horizontal_fov_width_cm = 2 * (np.tan(np.radians(horizontal_fov_degrees / 2)) * camera_height_cm)
vertical_fov_height_cm = 2 * (np.tan(np.radians(vertical_fov_degrees / 2)) * camera_height_cm)

# Calculate the cm_per_pixel ratio for both width and height
cm_per_pixel_width = horizontal_fov_width_cm / resolution_width
cm_per_pixel_height = vertical_fov_height_cm / resolution_height

# Box dimensions in pixels (from your image measurements)
box_width_px = 76.92  # Width in pixels

# Calculate the real-world dimensions of the box using both the width and height ratios
box_width_cm = box_width_px * cm_per_pixel_width
box_height_cm = box_width_px * cm_per_pixel_height  # if the real-world box is square

print(f"The box's real-world width is approximately: {box_width_cm:.2f} cm")
print(f"The box's real-world height is approximately: {box_height_cm:.2f} cm")
