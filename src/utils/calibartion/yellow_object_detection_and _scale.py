import cv2
import numpy as np

def detect_yellow_object(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image from BGR to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # Assuming the largest contour is the yellow object
    yellow_object_contour = max(yellow_contours, key=cv2.contourArea)

    return yellow_object_contour, image

def calculate_scale(contour, real_world_length):
    # Calculate the bounding rectangle for the contour
    x, y, width, height = cv2.boundingRect(contour)
    
    # Use the largest dimension of the object for scale calculation
    pixel_length = max(width, height)

    # Calculate the scale as real-world units per pixel
    scale = real_world_length / pixel_length

    return scale

# The known real-world length of the yellow object (in your chosen unit)
# This should be the length corresponding to the longest side of the object
real_world_length = 10 # for example, 10 centimeters

# Path to your image
image_path = 'path_to_your_image.jpg'

# Detect the yellow object in the image
yellow_object_contour, _ = detect_yellow_object(image_path)

# Calculate the scale
scale = calculate_scale(yellow_object_contour, real_world_length)

print(f"The scale is: {scale} real-world units per pixel")
