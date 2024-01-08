from math import sqrt, pow

# Defining a large value as infinity
INF = 10**18

# Function to return the Euclidean distance between two points
def dist(a, b):
    return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))

# Function to check whether a point lies inside or on the boundaries of the circle
def is_inside(c, p):
    return dist(c[0], p) <= c[1]

# Helper method to get a circle defined by 3 points
def get_circle_center(bx, by, cx, cy):
    B = bx * bx + by * by
    C = cx * cx + cy * cy
    D = bx * cy - by * cx
    return [(cy * B - by * C) / (2 * D), (bx * C - cx * B) / (2 * D)]

# Function to return a unique circle that intersects three points
def circle_from(A, B, C):
    I = get_circle_center(B[0] - A[0], B[1] - A[1], C[0] - A[0], C[1] - A[1])
    I[0] += A[0]
    I[1] += A[1]
    return [I, dist(I, A)]

# Function to return the smallest circle that intersects 2 points
def circle_from_two_points(A, B):
    C = [(A[0] + B[0]) / 2.0, (A[1] + B[1]) / 2.0]
    return [C, dist(A, B) / 2.0]

# Function to check whether a circle encloses the given points
def is_valid_circle(c, P):
    for p in P:
        if not is_inside(c, p):
            return False
    return True

# Function to find the minimum enclosing circle for a set of points
def minimum_enclosing_circle(P):
    n = len(P)
    if n == 0:
        return [[0, 0], 0]
    if n == 1:
        return [P[0], 0]

    mec = [[0, 0], INF]

    # Go over all pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            tmp = circle_from_two_points(P[i], P[j])
            if tmp[1] < mec[1] and is_valid_circle(tmp, P):
                mec = tmp

    # Go over all triples of points
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                tmp = circle_from(P[i], P[j], P[k])
                if tmp[1] < mec[1] and is_valid_circle(tmp, P):
                    mec = tmp

    return mec

# Function to convert normalized coordinates to pixel coordinates
def convert_to_pixel_coordinates(normalized_coords, image_width, image_height):
    pixel_coords = []
    for i in range(1, len(normalized_coords), 2):  # Skip the class label and iterate over pairs
        x_normalized = normalized_coords[i]
        y_normalized = normalized_coords[i + 1]
        x_pixel = int(x_normalized * image_width)
        y_pixel = int(y_normalized * image_height)
        pixel_coords.append((x_pixel, y_pixel))
    return pixel_coords

# Example YOLO segmentation output
yolo_output = [
    # Class label followed by pairs of normalized coordinates
    0, 0.0546875, 0.2875, 0.0421875, 0.290625, 0.0359375, 0.315625, 
    # ... more normalized coordinates ...
]

# Image dimensions (replace with actual dimensions of your images)
image_width = 640
image_height = 480

# Convert YOLO output to pixel coordinates
pixel_coordinates = convert_to_pixel_coordinates(yolo_output, image_width, image_height)

# Find the minimum enclosing circle
mec = minimum_enclosing_circle(pixel_coordinates)

# Print the result
if mec[0]:  # Check if a valid circle was found
    print("Center = { ", mec[0][0], ", ", mec[0][1], " } Radius = ", round(mec[1], 6))
else:
    print("No valid circle can be found (degenerate case).")
