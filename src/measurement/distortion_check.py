import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# Load the image


# Camera matrix
K = np.array([
    [2484.910468620379, 0, 2655.131772934547],
    [0, 2486.158169901567, 1470.720703865392],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients for fisheye (k1, k2, k3, k4)
D = np.array([0.0342138895741646, 0.0676732076535786, -0.0740896999695528, 0.029944425491756], dtype=np.float32)

# Get image dimensions

#save the undistorted image
# cv2.imwrite("C:/Users/gbo10/OneDrive/measurement_paper_images/for imageJ/check/GX010067_33_625.jpg_gamma_undistorted.jpg", undistorted_img)

#function that apply that distortion to a folder of images
def undistort_images(image_folder, K, D,undistorted_folder):
    # Get the list of image files in the folder
    image_files = os.listdir(image_folder)

    # Loop through each image file
    for image_file in tqdm(image_files):
        # Load the image
        image_path = os.path.join(image_folder, image_file)
        distorted_img = cv2.imread(image_path)

        # Get image dimensions
        h, w = distorted_img.shape[:2]

        # Compute the optimal new camera matrix
        new_camera_matrix = K.copy()
        new_size = (w, h)

        # Undistort the image
        undistorted_img = cv2.fisheye.undistortImage(distorted_img, K, D, Knew=new_camera_matrix, new_size=new_size)
        # Save the undistorted image
        #put in the undistorted folder
        undistorted_image_path = os.path.join(undistorted_folder, image_file.replace('.jpg', '_undistorted.jpg'))
        cv2.imwrite(undistorted_image_path, undistorted_img)
        

# Apply distortion correction to a folder of images
image_folder = "C:/Users/gbo10/OneDrive/measurement_paper_images/for imageJ/check"
undistort_images(image_folder, K, D, "C:/Users/gbo10/OneDrive/measurement_paper_images/for imageJ/carapace/1")



