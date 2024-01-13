import math
from enclosing_circle import minimum_enclosing_circle

def calculate_enclosing_diameter(points):
    # Function to calculate the diameter of the bounding circle based on the bounding box
    mec=minimum_enclosing_circle(points)

    return  round(mec[1], 6)*2


def convert_pixel_to_real_length(pixel_length):
    # Function to convert pixel length to real-world length


    scale_width = 5312 / 1366
    scale_height = 2988 / 768
    pixel_size = 0.00716844

    # Assuming the scaling factor is the same for width and height
    # If not, you need to calculate the width and height separately
    scale_factor = (scale_width + scale_height) / 2
    
    height_of_camera=650 # Height of camera in millimeters
    height_of_prawn=25

    distance_to_object =  height_of_camera-height_of_prawn# Distance to object in millimeters

    # Scale the pixel measurement back to original dimensions

    # Calculate the real-life width of the object
    real_width_cm = calculate_real_width(24.4 , distance_to_object, pixel_length*scale_factor, pixel_size)
    
    return real_width_cm


def calculate_real_width(focal_length, distance_to_object, width_in_pixels, pixel_size):
   
    # Calculate the width of the object in the image sensor plane in millimeters
    width_in_sensor = width_in_pixels * pixel_size

    # Calculate the real-life width of the object using the similar triangles principle
    real_width_mm = (width_in_sensor * distance_to_object) / focal_length

    # Convert the width from millimeters to centimeters
    real_width_cm = real_width_mm 

    return real_width_cm

   