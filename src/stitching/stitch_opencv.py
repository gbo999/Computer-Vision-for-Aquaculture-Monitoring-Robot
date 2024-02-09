import cv2
import glob
import os
import numpy as np

# List of image paths to be stitched
image_paths = glob.glob("src/stitching/output3/*.jpg")

# Load images
images = []
for path in image_paths:
    image = cv2.imread(path)
    scale_percent = 30 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    n = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    images.append(image)


def stitch_images(images):
    # Check if images list is empty
    if not images:
        print("No images provided for stitching!")
        return

    # Check if only one image is provided
    if len(images) == 1:
        print("Only one image provided for stitching!")
        return

    # Check if images have sufficient overlap
    # This is a simple check and may not work for all cases
    for i in range(len(images) - 1):
        diff = cv2.absdiff(images[i], images[i+1])
        non_zero_count = np.count_nonzero(diff)
        if non_zero_count < diff.size * 0.1:
            print(f"Images {i} and {i+1} do not have sufficient overlap!")
            return


import cv2
import numpy as np

def check_overlap(image1, image2):
    correlation = np.corrcoef(image1.ravel(), image2.ravel())[0, 1]
    print(f"Correlation between images: {correlation}")
    if correlation < 0.5:  # adjust this threshold as needed
        print(f"Possible insufficient overlap between images between {image1.file_name} and {image2.filename}")

def check_order(images):
    # assuming images are named in the format 'image1.jpg', 'image2.jpg', etc.
    image_numbers = [int(image.filename.split('image')[1].split('.jpg')[0]) for image in images]
    if image_numbers != sorted(image_numbers):
        print("Images might be in incorrect order")

def stitch_images(images):
    try:
        # Check if images list is empty
        if not images:
            print("No images provided for stitching!")
            return

        # Check if only one image is provided
        if len(images) == 1:
            print("Only one image provided for stitching!")
            return

        # Check if images have sufficient overlap
        for i in range(len(images) - 1):
            check_overlap(images[i], images[i+1])

        # Check if images are in the correct order
        check_order(images)

        stitcher = cv2.Stitcher_create()
        status, stitched_image = stitcher.stitch(images)
        
        if status == cv2.Stitcher_OK:
            output_path = "src/stitching/output3/stitched_image1.jpg"
            cv2.imwrite(output_path, stitched_image)
            print(f"Stitched image saved to {output_path}")
        elif status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("Insufficient images for stitching!")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Homography estimation failed!")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("Camera parameter adjustment failed!")
        else:
            print("Stitching failed!")
    except Exception as e:
        print(f"An error occurred: {e}")
# Stitch images
pano = stitch_images(images)

