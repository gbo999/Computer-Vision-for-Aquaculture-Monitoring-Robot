import cv2
import numpy as np

def draw_polygon(img, class_id, points):
    # Convert normalized coordinates to pixel coordinates
    img_height, img_width = img.shape[:2]
    pts = np.array([[int(x * img_width), int(y * img_height)] for x, y in points], np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw the polygon
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(img, str(class_id), tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return pts

def draw_diameter_line(img, points, focal_length, distance_to_object, pixel_size, original_size, resized_size):
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
            draw_diameter_line(img, pts,focal_length=24.4, distance_to_object=700, pixel_size=0.00716844, original_size=(5312, 2988), resized_size=(640, 360))

    return img
    # Display the image
    # cv2.imshow('Segmented Image with Diameters', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
# Example usage

import cv2
import numpy as np
from tkinter import *
from PIL import Image, ImageTk

# ... (Other function definitions remain unchanged) ...

# Initialize global variables for grid position and size
grid_x = 0
grid_y = 0
grid_size = 20  # Initial grid size
grid_angle = 0  # Initial grid rotation angle

def draw_grid_overlay(img, offset_x, offset_y, size, angle):
    # Create a new transparent overlay
    overlay = np.zeros_like(img, dtype=np.uint8)

    # Calculate rotation matrix
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    
    # Draw the grid lines on the overlay
    for i in range(0, rows, size):
        cv2.line(overlay, (0, i), (cols, i), (0, 255, 0), 1)
    for j in range(0, cols, size):
        cv2.line(overlay, (j, 0), (j, rows), (0, 255, 0), 1)

    # Apply rotation to the grid
    rotated_overlay = cv2.warpAffine(overlay, M, (cols, rows))
    
    # Apply the offset to the grid
    M_offset = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    translated_overlay = cv2.warpAffine(rotated_overlay, M_offset, (cols, rows))

    # Blend the overlay with the original image
    combined_img = cv2.addWeighted(img, 1, translated_overlay, 0.5, 0)

    return combined_img

def update_canvas(canvas, img, window):
    # Convert the processed image to PIL format and then to ImageTk format
    tk_img = ImageTk.PhotoImage(image=Image.fromarray(img))
    
    # If the canvas already has an image, remove it
    canvas.delete("all")
    
    # Add the new image to the canvas
    canvas.create_image(0, 0, image=tk_img, anchor=NW)
    canvas.image = tk_img  # Keep a reference!
    
    # Update the window
    window.update()

def main():
    # Process the image and get the result
    img = process_image('C:/Users/gbo10/Videos/research/counting_research_algorithms/src/to_colab/valid/images/GX010063_MP4-37_jpg.rf.1e299b123582106ea7e5baa9dd3cc866.jpg'
, 'C:/Users/gbo10/Dropbox/research videos/21.12/seg/zipfile (2)/content/runs/segment/predict/labels/GX010063_MP4-37_jpg.rf.1e299b123582106ea7e5baa9dd3cc866.txt')
    original_img = img.copy()

    # Create the main window and canvas
    window = Tk()
    window.title("Grid Overlay Interface")
    canvas = Canvas(window, width=img.shape[1], height=img.shape[0])
    canvas.pack()

    # Add sliders for grid control
    size_slider = Scale(window, from_=2, to=100, orient=HORIZONTAL, label="Grid Size")
    size_slider.pack()
    angle_slider = Scale(window, from_=-180, to=180, orient=HORIZONTAL, label="Grid Angle")
    angle_slider.pack()
    
    # Function to update the grid based on slider values
    def update_grid(event):
        global grid_size, grid_angle
        grid_size = size_slider.get()
        grid_angle = angle_slider.get()
        updated_img = draw_grid_overlay(original_img, grid_x, grid_y, grid_size, grid_angle)
        update_canvas(canvas, updated_img, window)

    # Bind the sliders to the update function
    size_slider.bind("<B1-Motion>", update_grid)
    angle_slider.bind("<B1-Motion>", update_grid)
    
    # Start with an initial grid
    update_grid(None)

    # Start the GUI event loop
    window.mainloop()

if __name__ == "__main__":
    main()
