import cv2
import numpy as np

# Checkerboard dimensions 
CHECKERBOARD = (4,3) 

# Sample checkerboard image
img = cv2.imread('C:/Users/gbo10/Videos/research/counting_research_algorithms/src/data/experiments/png/3.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# Find corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
print(ret)
# If corners are found, draw them
if ret:
    # Refine corner location based on pixel sub-corner area
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Draw the corners
    cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)

    # Show the image with drawn corners
    cv2.imshow('Chessboard Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the image with drawn corners
    