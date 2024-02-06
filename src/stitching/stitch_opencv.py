import cv2
import glob
import os

# List of image paths to be stitched
image_paths = glob.glob("src/stitching/output/*.jpg")

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
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        output_path = "src/stitching/output/stitched_image2.jpg"
        cv2.imwrite(output_path, stitched_image)
        print(f"Stitched image saved to {output_path}")
    else:
        print("Stitching failed!")


# Stitch images
pano = stitch_images(images)

