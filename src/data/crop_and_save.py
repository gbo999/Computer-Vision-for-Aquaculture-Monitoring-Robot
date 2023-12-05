import cv2
import pandas as pd
from tkinter import filedialog
from tkinter import Tk
import os
from datetime import datetime

def select_images():
    """ Open a dialog to select images """
    root = Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames()
    root.destroy()
    return list(file_paths)

def crop_image(image_path):
    """ Open an image and let the user crop it multiple times """
    image = cv2.imread(image_path)
    crops = []
    while True:
        cropped_area = cv2.selectROI("Image Cropper", image)
        cv2.destroyAllWindows()
        if cropped_area == (0, 0, 0, 0):
            break
        x, y, w, h = cropped_area
        crops.append(image[y:y+h, x:x+w])
    return crops

def generate_save_path(original_path, count):
    """ Generate a unique save path for each cropped image """
    directory, filename = os.path.split(original_path)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_filename = f"{name}_{count}_{timestamp}{ext}"
    return os.path.join(directory, 'cropped', new_filename)

def main():
    # Select images for cropping
    images = select_images()

    # Dataframe to store image info
    df = pd.DataFrame(columns=['File', 'Category'])

    for image_path in images:
        cropped_images = crop_image(image_path)
        count = 0
        for cropped in cropped_images:
            count += 1
            # Generate a new save path for each cropped image
            save_path = generate_save_path(image_path, count)
            cv2.imwrite(save_path, cropped)

            # Get category input from user
            category = input(f"Enter category for cropped image {count} from {image_path}: ")

            # Log data in dataframe
            df = df.append({'File': save_path, 'Category': category}, ignore_index=True)

    # Save the dataframe to CSV
    df.to_csv('image_data.csv', index=False)

if __name__ == "__main__":
    main()
