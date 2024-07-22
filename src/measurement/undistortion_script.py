import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the distorted image
image_path = "C:/Users/gbo10/OneDrive/measurement_paper_images/validation/GX010191_20_455.jpg"
distorted_img = cv2.imread(image_path)

# Camera matrix
K = np.array([
    [2484.910468620379, 0, 2655.131772934547],
    [0, 2486.158169901567, 1470.720703865392],
    [0, 0, 1]
], dtype=np.float32)

# Original distortion coefficients for fisheye (k1, k2, k3, k4)
original_D = np.array([0.0342138895741646, 0.0676732076535786, -0.0740896999695528, 0.029944425491756], dtype=np.float32)

# Scaling factor for distortion (0 = no undistortion, 1 = full undistortion)
lens_correction_amount = 0.5  # Adjust this value between 0 and 1 to control the amount of undistortion

# Calculate the correction factor
factor = max(1.0 - lens_correction_amount, 0.001)

# Adjust the focal length based on the correction factor
out_c = (K[0, 2], K[1, 2])  # Optical center
out_f = (K[0, 0] / factor, K[1, 1] / factor)  # Focal lengths

# Get image dimensions
h, w = distorted_img.shape[:2]

# Initialize the undistort rectify map with the adjusted focal length
new_camera_matrix = K.copy()
new_camera_matrix[0, 0] = out_f[0]
new_camera_matrix[1, 1] = out_f[1]

# Function to apply the Rust-like calculations for point correction
def correct_point(point, out_c, out_f, params):
    new_out_pos = point
    
    # Simulate digital lens undistortion if applicable (placeholder)
    # if has_digital_lens:
    #     new_out_pos = digital_lens_undistort(new_out_pos, params)
    
    # Scale and center the point
    new_out_pos = ((new_out_pos[0] - out_c[0]) / out_f[0], (new_out_pos[1] - out_c[1]) / out_f[1])
    
    # Apply distortion model (placeholder for actual distortion model logic)
    # new_out_pos = distortion_model_undistort(new_out_pos, params)
    
    # Correct for light refraction
    if params['light_refraction_coefficient'] != 1.0 and params['light_refraction_coefficient'] > 0.0:
        r = np.sqrt(new_out_pos[0]**2 + new_out_pos[1]**2)
        if r != 0.0:
            sin_theta_d = (r / np.sqrt(1.0 + r * r)) / params['light_refraction_coefficient']
            r_d = sin_theta_d / np.sqrt(1.0 - sin_theta_d * sin_theta_d)
            factor = r_d / r
            new_out_pos = (new_out_pos[0] * factor, new_out_pos[1] * factor)
    
    # Rescale and recenter the point
    new_out_pos = ((new_out_pos[0] * out_f[0]) + out_c[0], (new_out_pos[1] * out_f[1]) + out_c[1])
    
    # Blend the corrected point with the original point
    out_pos = (
        new_out_pos[0] * (1.0 - params['lens_correction_amount']) + (point[0] * params['lens_correction_amount']),
        new_out_pos[1] * (1.0 - params['lens_correction_amount']) + (point[1] * params['lens_correction_amount']),
    )
    
    return out_pos

# Parameters for correction
params = {
    'lens_correction_amount': lens_correction_amount,
    'light_refraction_coefficient': 1.33 # Adjust as needed
}

# Apply the point correction to the entire image
undistorted_img = np.zeros_like(distorted_img)
for i in range(h):
    for j in range(w):
        undistorted_point = correct_point((j, i), out_c, out_f, params)
        x, y = int(undistorted_point[0]), int(undistorted_point[1])
        if 0 <= x < w and 0 <= y < h:
            undistorted_img[y, x] = distorted_img[i, j]

# Display the original and undistorted images
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Original image
axs[0].imshow(cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Distorted Image')
axs[0].axis('on')

# Undistorted image
axs[1].imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
axs[1].set_title('Undistorted Image')
axs[1].axis('on')

plt.show()
