import numpy as np

# Camera height from the floor in cm and FOV in degrees
camera_height_cm = 73.5
vertical_fov_deg = 55.9

# Calculate the vertical FOV in radians
vertical_fov_rad = np.radians(vertical_fov_deg)

# Image resolution in pixels
resolution_width_px = 1600
resolution_height_px = 1200

# Calculate the viewable height at the floor level based on the vertical FOV
viewable_height_cm = 2 * (camera_height_cm * np.tan(vertical_fov_rad / 2))

# The aspect ratio (width:height) of the image is 3:4
aspect_ratio = 3 / 4

# Calculate the viewable width using the aspect ratio
# Divide the viewable height by the aspect ratio to find the corresponding width
viewable_width_cm = viewable_height_cm / aspect_ratio

# Calculate the width and height in cm that each pixel represents
cm_per_pixel_width = viewable_width_cm / resolution_width_px
cm_per_pixel_height = viewable_height_cm / resolution_height_px

# Box dimensions in pixels (assuming the box is square, the width and height will be the same)
box_dimension_px = 76.92

# Calculate the real-world dimensions of the box using the ratios
box_width_cm = box_dimension_px * cm_per_pixel_width
box_height_cm = box_dimension_px * cm_per_pixel_height

# Print the calculated real-world dimensions of the box
print(f"The box's real-world width is approximately: {box_width_cm:.2f} cm")
print(f"The box's real-world height is approximately: {box_height_cm:.2f} cm")
