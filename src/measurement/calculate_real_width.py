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




# Example usage
focal_length = 24.4  # Focal length in millimetersi
distance_to_object =700 # Distance to object in millimeters
width_in_pixels =696.7 # Width of the object in pixels
pixel_size =0.00716844   # Pixel size in millimeters (0.8 micrometers)

# Calculate the real-life width
real_width = calculate_real_width(focal_length, distance_to_object, width_in_pixels, pixel_size)
print(f"The real-life width of the object is {real_width} mm.")

#https://pixelcalculator.com/en