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

# Rest of your script including the 'main' function
# ...

if __name__ == "__main__":
    main()
