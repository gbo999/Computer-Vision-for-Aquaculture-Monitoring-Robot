import cv2
import os
import numpy as np
from enclosing_circle import minimum_enclosing_circle

class ImageViewer:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.current_image_index = 0
        self.current_segmentation_index = 0
        self.images = self._load_images()
        self.labels = self._load_labels()

    def _load_images(self):
        images = [img for img in sorted(os.listdir(self.image_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
        print( f'len {len(images)}')
        return images

    def _load_labels(self):
        labels = [label for label in sorted(os.listdir(self.label_dir)) if label.endswith('.txt')]
        return labels

    def show_image(self):
        if self.current_image_index < len(self.images):
            print(f'current_image_index {self.current_image_index}')
            # Reload the image to clear previous annotations
            image_path = os.path.join(self.image_dir, self.images[self.current_image_index])
            image = cv2.imread(image_path)
            self._draw_annotations(image)
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _draw_annotations(self, image):
        label_path = os.path.join(self.label_dir, self.labels[self.current_image_index])
        with open(label_path, 'r') as file:
           for line in file:
            parts = line.split()
            points = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
            self._draw_single_annotation(image, points)

    def _draw_single_annotation(self, image, points):
        if len(points) > 2:
            pixel_points = self._convert_to_pixels(points, image.shape[1], image.shape[0])
            cv2.polylines(image, [np.array(pixel_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            center, radius = minimum_enclosing_circle(pixel_points)
            if center and radius:
                cv2.circle(image, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)

    def _convert_to_pixels(self, normalized_points, image_width, image_height):
        pixel_points = []
        print(f'len {len(normalized_points)}')
        for x_normalized, y_normalized in normalized_points:
            x_pixel = int(x_normalized * image_width)
            y_pixel = int(y_normalized * image_height)
            pixel_points.append((x_pixel, y_pixel))
        return pixel_points

    def navigate_images(self, key):
        if key == 'd' and self.current_image_index < len(self.images) - 1:
            print(f'key was pressed {key}')
            self.current_image_index += 1
            self.current_segmentation_index = 0  # Reset segmentation index
        elif key == 'a' and self.current_image_index > 0:
            self.current_image_index -= 1
            self.current_segmentation_index = 0  # Reset segmentation index

    def navigate_segmentations(self, key):
        if key == 'z' and self.current_segmentation_index > 0:
            self.current_segmentation_index -= 1
            self.show_image()  # Refresh the image with the new segmentation
        elif key == 'c' and self.current_segmentation_index < len(self.labels) - 1:
            self.current_segmentation_index += 1
            self.show_image()  


    def run_viewer(self):
        import keyboard  # Make sure to import the keyboard library

        while True:
            self.show_image()

            # Wait for a key press
            key = None
            while True:
                if keyboard.is_pressed('d'):
                    key = 'd'
                    break
                elif keyboard.is_pressed('a'):
                    key = 'a'
                    break
                elif keyboard.is_pressed('c'):
                    key = 'c'
                    break
                elif keyboard.is_pressed('z'):
                    key = 'z'
                    break
                elif keyboard.is_pressed('q'):  # Using 'q' to quit
                    key = 'q'
                    break

            if key == 'd':
                self.navigate_images('d')
            elif key == 'a':
                self.navigate_images('a')
            elif key == 'c':
                self.navigate_segmentations('c')
            elif key == 'z':
                self.navigate_segmentations('z')
            elif key == 'q':
                break

            cv2.destroyAllWindows()

