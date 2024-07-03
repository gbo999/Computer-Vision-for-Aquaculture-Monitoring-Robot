import cv2
import numpy as np

def draw_polygon(img, class_id, points):
    """
    Draws a polygon on the given image.

    Args:
        img (numpy.ndarray): The image on which the polygon will be drawn.
        class_id (int): The class ID associated with the polygon.
        points (list): A list of normalized coordinates of the polygon vertices.

    Returns:
        numpy.ndarray: The reshaped array of polygon vertices.

    """
    # Convert normalized coordinates to pixel coordinates
    img_height, img_width = img.shape[:2]
    pts = np.array([[int(x * img_width), int(y * img_height)] for x, y in points], np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw the polygon
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(img, str(class_id), tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return pts

import cv2
import numpy as np

def draw_diameter_line(img, points, focal_length, distance_to_object, pixel_size, original_size, resized_size):
    """
    Draws a diameter line on the given image and calculates the real-life width of the object.

    Args:
        img (numpy.ndarray): The image on which to draw the diameter line.
        points (list): A list of points representing the segmentation.
        focal_length (float): The focal length of the camera in pixels.
        distance_to_object (float): The distance from the camera to the object in centimeters.
        pixel_size (float): The size of a pixel in millimeters.
        original_size (tuple): The original dimensions of the image (width, height).
        resized_size (tuple): The resized dimensions of the image (width, height).

    Returns:
        None
    """
    # Find the two farthest points in the segmentation
    max_distance = 0
    point1 = points[0][0]
    point2 = points[0][0]

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i][0]
            p2 = points[j][0]
            distance = np.linalg.norm(p1 - p2)
            if distance > max_distance:
                max_distance = distance
                point1 = p1
                point2 = p2

    # Draw the diameter line
    cv2.line(img, tuple(point1), tuple(point2), (0, 0, 255), 2)
    
    # Calculate the scaling factor due to image resizing
    scale_width = original_size[0] / resized_size[0]
    scale_height = original_size[1] / resized_size[1]
    length_in_squares = max_distance / 6

    # Assuming the scaling factor is the same for width and height
    # If not, you need to calculate the width and height separately
    scale_factor = (scale_width + scale_height) / 2
    
    # Scale the pixel measurement back to original dimensions
    width_in_original_pixels = int(max_distance * scale_factor)

    # Calculate the real-life width of the object
    real_width_cm = calculate_real_width(focal_length, distance_to_object, width_in_original_pixels, pixel_size)
    
    # Draw the real-life width as text on the image
    text_position = (min(point1[0], point2[0]), min(point1[1], point2[1]) - 10)  # Adjust text position as needed
    cv2.putText(img, f"{real_width_cm:.2f} mm", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    text_position_squares = (text_position[0], text_position[1] - 20)
    cv2.putText(img, f"{length_in_squares:.2f} squares", text_position_squares, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

 # Resized image size (width, height)

def calculate_real_width(focal_length, distance_to_object, width_in_pixels, pixel_size):
    """
    Calculate the real-life width of an object.

    Parameters:
    focal_length (float): Focal length of the camera lens in millimeters (mm).
    distance_to_object (float): Distance from the camera to the object in millimeters (mm).
    width_in_pixels (int): Width of the object in pixels on the image sensor.
    pixel_size (float): Size of a pixel on the image sensor in millimeters (mm).

    Returns:
    float: Real-life width of the object in centimeters (cm).
    """
    # Calculate the width of the object in the image sensor plane in millimeters
    width_in_sensor = width_in_pixels * pixel_size

    # Calculate the real-life width of the object using the similar triangles principle
    real_width_mm = (width_in_sensor * distance_to_object) / focal_length

    # Convert the width from millimeters to centimeters
    real_width_cm = real_width_mm 

    return real_width_cm
def diameter_length_in_squares(points, square_size_pixels):
    # Find the two farthest points in the segmentation
    max_distance = 0
    point1 = points[0][0]
    point2 = points[0][0]

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i][0]
            p2 = points[j][0]
            distance = np.linalg.norm(p1 - p2)
            if distance > max_distance:
                max_distance = distance
                point1 = p1
                point2 = p2

    # Calculate the length of the diameter in squares
    length_in_squares = max_distance / square_size_pixels
    return length_in_squares
    
    return covered_squares
def process_image(image_path, label_path):
    # Load the image and resize
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    img = cv2.resize(img, (640, 360))

    # Read the YOLO label file
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.split()
            class_id = int(parts[0])
            points = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
            pts = draw_polygon(img, class_id, points)
            draw_diameter_line(img, pts,focal_length=24.4, distance_to_object=670, pixel_size=0.00716844, original_size=(5312, 2988), resized_size=(640, 360))
            square_size_pixels = 6 # size of a 10x10 mm square in pixels
# Calculate the length of the diameter in squares
            length_in_squares = diameter_length_in_squares(pts, square_size_pixels)
            print(f"The diameter is approximately {length_in_squares:.2f} squares long.")


    # Display the image
    cv2.imshow('Segmented Image with Diameters', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Example usage
process_image('C:/Users/gbo10/Videos/research/counting_research_algorithms/src/to_colab/valid/images/GX010063_MP4-37_jpg.rf.1e299b123582106ea7e5baa9dd3cc866.jpg'
, 'C:/Users/gbo10/Dropbox/research videos/21.12/seg/zipfile (2)/content/runs/segment/predict/labels/GX010063_MP4-37_jpg.rf.1e299b123582106ea7e5baa9dd3cc866.txt')



# You would call this function with your list of polygons and the size of a square in pixels
# This requires you to calculate 'square_size_pixels' based on your image scale
