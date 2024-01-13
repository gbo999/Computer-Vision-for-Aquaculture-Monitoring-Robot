import matplotlib.pyplot as plt
import numpy as np

# Parameters
camera_height_cm = 73.5
vertical_fov_deg = 55.9
vertical_fov_rad = np.radians(vertical_fov_deg)

# Calculating the viewable distance on the ground (one side)
ground_distance_cm = camera_height_cm * np.tan(vertical_fov_rad / 2)

# Plot
fig, ax = plt.subplots()

# Camera
ax.plot([0, 0], [0, camera_height_cm], 'ro-')  # Camera line
ax.text(-5, camera_height_cm / 2, 'Camera height', rotation='vertical', verticalalignment='center')

# Ground line
ax.plot([-ground_distance_cm, ground_distance_cm], [0, 0], 'k-', lw=2)

# FOV lines
ax.plot([0, ground_distance_cm], [camera_height_cm, 0], 'b--')  # Right FOV line
ax.plot([0, -ground_distance_cm], [camera_height_cm, 0], 'b--')  # Left FOV line

# Annotations
ax.text(ground_distance_cm / 2, -10, 'Ground distance', horizontalalignment='center')
ax.text(ground_distance_cm + 10, camera_height_cm / 2, 'FOV line', rotation=-45)
ax.text(0, camera_height_cm + 3, 'Camera', horizontalalignment='center')

# Setting limits and labels
ax.set_xlim(-ground_distance_cm * 1.5, ground_distance_cm * 1.5)
ax.set_ylim(0, camera_height_cm * 1.5)
ax.set_xlabel('Distance (cm)')
ax.set_ylabel('Height (cm)')
ax.set_title('Camera Field of View Illustration')

# Show plot
plt.show()
