import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# Load the image


# Camera matrix
K = np.array([
     [ 2404.59244190324, 0.0,              1919.372365976781 ],
      [ 0.0,                2405.799814606758, 1063.171593155705 ],
      [ 0.0,                0.0,              1.0 ]
], dtype=np.float32)

# Distortion coefficients for fisheye (k1, k2, k3, k4)
D = np.array([       -0.2003271,   2.91272175, -6.56042688,  5.60476002

], dtype=np.float32)

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

        


        undistorted_img = cv2.fisheye.undistortImage(distorted_img, K, D, Knew=new_camera_matrix, new_size=new_size)
        # Save the undistorted image
        #put in the undistorted folder
        undistorted_image_path = os.path.join(undistorted_folder, image_file.replace('.jpg', '_undistorted.jpg'))
        cv2.imwrite(undistorted_image_path, undistorted_img)
        # Display the undistorted image
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(distorted_img, cv2.COLOR_BGR2RGB))
        plt.title("Distorted Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
        plt.title("Undistorted Image")
        plt.axis("off")

#


# Apply distortion correction to a folder of images
image_folder = "C:/Users/gbo10/OneDrive/measurement_paper_images/for imageJ/full body/1"
undistort_images(image_folder, K, D, "C:/Users/gbo10/OneDrive/measurement_paper_images/for imageJ/carapace/1")



