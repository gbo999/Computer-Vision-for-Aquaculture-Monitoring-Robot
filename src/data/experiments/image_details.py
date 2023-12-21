from PIL import Image
import os

def analyze_image(image_path, dpi=300):
    # Open the image
    with Image.open(image_path) as img:
        width, height = img.size  # Image dimensions in pixels
        color_depth = img.mode    # Color depth (e.g., RGB, RGBA, etc.)

    # Calculate values
    pixel_total = width * height
    picture_size = (width / dpi, height / dpi)  # In inches
    disk_space = os.path.getsize(image_path)    # File size in bytes
    pixel_size = 25.4 / dpi  # Pixel size in mm (assuming 25.4 mm/inch)

    return {
        "Pixel Total": pixel_total,
        "Picture Size (inches)": picture_size,
        "DPI/PPI": dpi,
        "Color Depth": color_depth,
        "Disk Space (bytes)": disk_space,
        "Pixel Size (mm)": pixel_size
    }

# Example usage
info = analyze_image("../experiments/15.jpg")
for key, value in info.items():
    print(f"{key}: {value}")
