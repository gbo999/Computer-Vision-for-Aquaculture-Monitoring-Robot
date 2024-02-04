import cv2
import glob

# List of image paths to be stitched
image_paths = glob.glob("src/stitching/output/*.jpg")

# Load images
images = []
for path in image_paths:
    image = cv2.imread(path)
    images.append(image)

def stitch_images(images):
    stitcher = cv2.Stitcher_create ()
    status, stitched_image = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        cv2.imshow("Stitched Image", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed!")


# Stitch images
pano=stitch_images(images)

