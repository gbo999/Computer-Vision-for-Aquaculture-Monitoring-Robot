from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
import pandas as pd
import csv

class AbstractImageViewer(ABC):
    def __init__(self, image_dir, label_dir, imagej_results_file):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = self._load_images()
        self.current_image_name = self.images[0] if self.images else None
        self.imagej_results = self._load_imagej_results(imagej_results_file)

    def _load_images(self):
        return [img for img in sorted(os.listdir(self.image_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def _load_imagej_results(self, imagej_results_file):
        return pd.read_csv(imagej_results_file)

    @abstractmethod
    def _load_labels(self, image_name):
        pass

    @abstractmethod
    def _draw_annotations(self, image):
        pass

    def resize_image_to_screen(self, image, screen_width, screen_height, keep_aspect_ratio=True):
        img_height, img_width = image.shape[:2]
        if keep_aspect_ratio:
            width_ratio = screen_width / img_width
            height_ratio = screen_height / img_height
            resize_ratio = min(width_ratio, height_ratio)
            new_width = int(img_width * resize_ratio)
            new_height = int(img_height * resize_ratio)
            image = cv2.resize(image, (new_width, new_height))
        else:
            image = cv2.resize(image, (screen_width, screen_height))
        return image

    def show_image(self):
        if self.current_image_name:
            image_path = os.path.join(self.image_dir, self.current_image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return
            screen_width, screen_height = 5312, 2988
            resized_image = self.resize_image_to_screen(image, screen_width, screen_height, keep_aspect_ratio=False)
            self._draw_annotations(resized_image)
            cv2.imshow('Image', resized_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def navigate_images(self, key):
        current_index = self.images.index(self.current_image_name)
        if key == 'd' and current_index < len(self.images) - 1:
            self.current_image_name = self.images[current_index + 1]
        elif key == 'a' and current_index > 0:
            self.current_image_name = self.images[current_index - 1]

    def run_viewer(self):
        import keyboard  # Make sure to import the keyboard library
        while True:
            self.show_image()
            while True:
                if keyboard.is_pressed('d'):
                    self.navigate_images('d')
                    break
                elif keyboard.is_pressed('a'):
                    self.navigate_images('a')
                    break
                elif keyboard.is_pressed('q'):
                    cv2.destroyAllWindows()  # Using 'q' to quit
                    break
            # Continue displaying images
