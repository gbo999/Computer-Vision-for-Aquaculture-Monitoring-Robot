import cv2
import os
from measurements_calculator import calculate_enclosing_diameter
import numpy as np

class ImageViewer:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.current_index = 0
        self.images = self._load_images()
        self.labels = self._load_labels()

    def _load_images(self):
        images = [img for img in sorted(os.listdir(self.image_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]
        return images

    def _load_labels(self):
        labels = [label for label in sorted(os.listdir(self.label_dir)) if label.endswith('.txt')]
        return labels

    def show_image(self):
        if self.current_index < len(self.images):
            image_path = os.path.join(self.image_dir, self.images[self.current_index])
            image = cv2.imread(image_path)
            self._draw_annotations(image)
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _draw_annotations(self, image):
        label_path = os.path.join(self.label_dir, self.labels[self.current_index])
        with open(label_path, 'r') as file:
            for line in file:
                # Extracting the segmentation points
                parts = line.strip().split()
                class_id = parts[0]
                normalized_points = [float(p) for p in parts[1:]]
                
                # Convert normalized values to pixel values
                pixel_points = self._convert_to_pixels(normalized_points, image.shape[1], image.shape[0])
                
                # Check if there are enough points to form a contour
                if len(pixel_points) > 2:
                    # Draw the segmentation mask (contour)
                    cv2.polylines(image, [np.array(pixel_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                    # Calculate and draw the minimum enclosing circle
                    center, radius = calculate_enclosing_diameter(pixel_points)
                    if center and radius:  # Valid circle
                        cv2.circle(image, (int(center[0]), int(center[1])), int(radius), (0, 0, 255), 2)


    def _convert_to_pixels(self, normalized_points, image_width, image_height):
        pixel_points = []
        for i in range(0, len(normalized_points), 2):
            x_pixel = int(normalized_points[i] * image_width)
            y_pixel = int(normalized_points[i + 1] * image_height)
            pixel_points.append((x_pixel, y_pixel))
        return pixel_points

    def navigate_images(self, key):
        if key == 'd' and self.current_index < len(self.images) - 1:
            self.current_index += 1
        elif key == 'a' and self.current_index > 0:
            self.current_index -= 1

    def run_viewer(self):
        # Main loop to capture key presses and show images
        pass
