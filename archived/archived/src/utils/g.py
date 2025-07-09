import numpy as np
import imageio
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
class FisheyeCalibrator:
    def __init__(self):
        self.map1 = None
        self.map2 = None

    def load_stmap(self, filename):
        # Load the image file
        img = imageio.imread(filename)
        
        # Separate the channels
        red, green, blue = img[:,:,0], img[:,:,1], img[:,:,2]
        
        # Reconstruct the maps
        self.map1 = red * self.calib_dimension[0]
        self.map2 = green * self.calib_dimension[1]

        # Ensure maps are of type CV_32FC1
        self.map1 = self.map1.astype(np.float32)
        self.map2 = self.map2.astype(np.float32)

        return self.map1, self.map2

    def set_calib_dimension(self, calib_dimension):
        self.calib_dimension = calib_dimension
def undistort_image(image_filename, map1, map2):
    # Load the distorted image
    distorted_img = cv2.imread(image_filename)

    # Perform the remapping
    undistorted_img = cv2.remap(distorted_img, map1, map2, interpolation=cv2.INTER_LINEAR)

    return undistorted_img
if __name__ == "__main__":
    Tk().withdraw()  # Hide root window

    # Prompt the user to select the STMap file
    stmap_filename = askopenfilename(title="Select STMap file",
                                     filetypes=((".exr", "*.exr"), (".png", "*.png"),))

    if not stmap_filename:
        print("No STMap file selected")
        exit()

    # Prompt the user to select the distorted image file
    distorted_image_filename = askopenfilename(title="Select distorted image file",
                                               filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp"),))

    if not distorted_image_filename:
        print("No image file selected")
        exit()

    # Initialize the FisheyeCalibrator and set the calibration dimensions
    calib_dimension = (5312, 2988)  # Replace with the actual dimensions used during calibration
    fisheye_calibrator = FisheyeCalibrator()
    fisheye_calibrator.set_calib_dimension(calib_dimension)

    # Load the STMap
    map1, map2 = fisheye_calibrator.load_stmap(stmap_filename)

    # Undistort the image
    undistorted_img = undistort_image(distorted_image_filename, map1, map2)

    # Prompt the user to save the undistorted image
    save_path = asksaveasfilename(title="Save undistorted image",
                                  filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("BMP files", "*.bmp")),
                                  defaultextension=".png")

    if save_path:
        cv2.imwrite(save_path, undistorted_img)
        print(f"Undistorted image saved at {save_path}")
    else:
        print("No save location provided")
