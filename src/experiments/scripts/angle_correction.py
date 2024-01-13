import math


measured_height_px=0

# Assume we have measured the tilt angle in degrees.
tilt_angle_degrees = ...  # Angle of tilt in degrees

# Convert the angle to radians for computation.
tilt_angle_radians = math.radians(tilt_angle_degrees)

# Compute the correction factor.
# This is valid for small tilts and when the tilt is in one plane only.
correction_factor = 1 / math.cos(tilt_angle_radians)

# Apply the correction factor to the measured pixel height.
corrected_height_px = measured_height_px * correction_factor
