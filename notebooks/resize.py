import os
from PIL import Image

def resize_images(input_folder, output_folder, size):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Open the image
            image = Image.open(os.path.join(input_folder, filename))

            # Resize the image
            resized_image = image.resize(size)

            # Save the resized image to the output folder
            resized_image.save(os.path.join(output_folder, filename))

            # Close the image
            image.close()

# Usage example
input_folder = "C:/Users/gbo10/OneDrive/pictures/to_contrast/2.1/gamma"
output_folder = "C:/Users/gbo10/OneDrive/pictures/to_contrast/2.1/gamma/resized"
size = (640, 640)

resize_images(input_folder, output_folder, size)