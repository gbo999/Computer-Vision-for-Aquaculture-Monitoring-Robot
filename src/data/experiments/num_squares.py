import cv2
import numpy as np

# Load the image
image = cv2.imread('path_to_your_image.png')

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a blur to the grayscale image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Threshold the image to get a binary image of the prawn
_, thresholded_image = cv2.threshold(blurred_image, your_threshold_value, 255, cv2.THRESH_BINARY)

# Perform morphological operations
kernel = np.ones((5, 5), np.uint8)
cleaned_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

# Find contours of the prawn
contours, _ = cv2.findContours(cleaned_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour is the prawn
prawn_contour = max(contours, key=cv2.contourArea)

# Calculate the bounding rectangle around the prawn
x, y, w, h = cv2.boundingRect(prawn_contour)

# Assume you know the size of a grid square (e.g., grid_square_size in cm or mm)
grid_square_size = your_grid_square_size

# Estimate the number of squares covered
num_squares_width = w / grid_square_size
num_squares_height = h / grid_square_size

# Assuming the prawn is approximately rectangular and covers a full grid square
num_squares_covered = int(round(num_squares_width)) * int(round(num_squares_height))

print(f"The prawn covers approximately {num_squares_covered} squares.")

# Display the result
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Prawn', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
