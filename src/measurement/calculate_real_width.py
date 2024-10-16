def calculate_real_width(focal_length, distance_to_object, width_in_pixels, pixel_size):
    """
    Calculate the real-life width of an object.

    Parameters:
    focal_length (float): Focal length of the camera lens in millimeters (mm).
    distance_to_object (float): Distance from the camera to the object in millimeters (mm).
    width_in_pixels (int): Width of the object in pixels on the image sensor.
    pixel_size (float): Size of a pixel on the image sensor in millimeters (mm).

    Returns:
    float: Real-life width of the object in centimeters (cm)
     * pixel_size.
    """
    # Calculate the width of the object in the image sensor plane in millimeters
    width_in_sensor = width_in_pixels* pixel_size

    # Calculate the real-life width of the object using the similar triangles principle
    real_width_mm = (width_in_sensor * distance_to_object) / focal_length

    # Convert the width from millimeters to centimeters
    

    return real_width_mm



def calculate_distance(focal_length, real_width, width_in_pixels, pixel_size):
    """
    Calculate the distance to an object.

    Parameters:
    focal_length (float): Focal length of the camera lens in millimeters (mm).
    real_width (float): Real-life width of the object in centimeters (cm).
    width_in_pixels (int): Width of the object in pixels on the image sensor.
    pixel_size (float): Size of a pixel on the image sensor in millimeters (mm).

    Returns:
    float: Distance from the camera to the object in millimeters (mm).
    """
    # Calculate the width of the object in the image sensor plane in millimeters
    width_in_sensor = width_in_pixels * pixel_size

    # Calculate the distance to the object using the similar triangles principle
    distance_mm = (real_width * focal_length) / width_in_sensor

    return distance_mm
def calculate_real_width_foc(focal_length_pixels, distance_to_object, width_in_pixels):
    """
    Calculate the real-life width of an object using focal length in pixels.

    Parameters:
    focal_length_pixels (float): Focal length of the camera lens in pixels.
    distance_to_object (float): Distance from the camera to the object in millimeters (mm).
    width_in_pixels (int): Width of the object in pixels on the image sensor.

    Returns:
    float: Real-life width of the object in millimeters (mm)
    """
    # Calculate the real-life width of the object using the similar triangles principle
    real_width_mm = (width_in_pixels * distance_to_object) / focal_length_pixels

    return real_width_mm

print(calculate_real_width_foc(3846.69, 650, 55))

# Example usage
focal_length =24.22


distance_to_object =650
width_in_pixels =711
pixel_size =0.00716844 
# Calculate the real-life width
real_width = calculate_real_width(focal_length, distance_to_object, width_in_pixels, pixel_size)
print(f"The real-life width of the object is {real_width} mm.")

#https://pixelcalculator.com/en


3846.69

3574.91

import math

# Function to calculate focal length based on AFOV (angular field of view) and height (h)
import math

def calculate_focal_length(fov_degrees, sensor_width_mm):
    """
    Calculate the focal length using the field of view and sensor width.

    Parameters:
    fov_degrees (float): Horizontal field of view in degrees.
    sensor_width_mm (float): Width of the sensor in millimeters.

    Returns:
    float: Focal length in millimeters.
    """
    # Convert field of view from degrees to radians
    fov_radians = math.radians(fov_degrees)
    
    # Calculate focal length
    focal_length_mm = (sensor_width_mm / 2) / math.tan(fov_radians / 2)
    
    return focal_length_mm

# Example usage:
fov = 76.2  # Field of view in degrees
sensor_width =11.65  # Sensor width in mm (for 1/1.9" sensor)

focal_length = calculate_focal_length(fov, sensor_width)
print(f"Calculated focal length: {focal_length:.2f} mm")

# Example usage:
AFOV_example = 76.2  # in degrees
h_example = 11.65  # in some units (e.g., mm)

focal_length_result = calculate_focal_length(AFOV_example, h_example)
print(f"The calculated focal length is {focal_length_result} units.")
