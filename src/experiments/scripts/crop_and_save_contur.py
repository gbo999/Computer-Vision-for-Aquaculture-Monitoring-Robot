import cv2
import pandas as pd
import numpy as np
from tkinter import filedialog
from tkinter import Tk
import os
from datetime import datetime

def segment_prawns(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmented_prawns = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust this threshold as needed
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Apply the mask to the original image
            result = cv2.bitwise_and(image, image, mask=mask)
            
            # Get bounding box coordinates to crop the masked image
            x, y, w, h = cv2.boundingRect(contour)
            cropped = result[y:y+h, x:x+w]
            segmented_prawns.append(cropped)

    return segmented_prawns

def generate_save_path(original_path, count):
    directory, filename = os.path.split(original_path)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_filename = f"{name}_{count}_{timestamp}{ext}"
    return os.path.join(directory, 'cropped', new_filename)

def main():
    images = select_images()
    df = pd.DataFrame(columns=['File', 'Category'])

    for image_path in images:
        image = cv2.imread(image_path)
        cropped_images = segment_prawns(image)
        count = 0
        for cropped in cropped_images:
            count += 1
            save_path = generate_save_path(image_path, count)
            cv2.imwrite(save_path, cropped)
            category = input(f"Enter category for segmented prawn {count} from {image_path}: ")
            df = df.append({'File': save_path, 'Category': category}, ignore_index=True)

    df.to_csv('image_data.csv', index=False)

if __name__ == "__main__":
    main()