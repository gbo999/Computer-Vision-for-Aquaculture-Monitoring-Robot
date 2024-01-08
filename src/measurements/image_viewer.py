import cv2
import os
import numpy as np
from enclosing_circle import minimum_enclosing_circle
from measurements_calculator import convert_pixel_to_real_length
import csv

class ImageViewer:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.current_image_index = 0
        self.current_segmentation_index = 0
        self.images = self._load_images()
        self.labels = self._load_labels()
        self.length_by_calc = 0

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
            lines = file.readlines()

            if self.current_segmentation_index < len(lines):
                line = lines[self.current_segmentation_index]
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
                self.length_by_calc = round(radius*2, 6)


            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 255)  # White color
            line_type = 2
            # Position for the text - Adjust as needed
            text_pos = (int(center[0] + radius), int(center[1]))
            # Put the diameter text on the image
            cv2.putText(image, 
                        f"Diameter: {self.length_by_calc}", 
                        text_pos, 
                        font, 
                        font_scale, 
                        font_color, 
                        line_type)
            text_pos2 = (int(center[0] ), int(center[1]+30))
            # Put the diameter text on the image
            cv2.putText(image, 
                        f"Diameter_real: {convert_pixel_to_real_length(self.length_by_calc)}", 
                        text_pos2, 
                        font, 
                        font_scale, 
                        font_color, 
                        line_type)

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
             # Refresh the image with the new segmentation
        elif key == 'v' and self.current_segmentation_index < len(self.labels) - 1:
            self.current_segmentation_index += 1 

    def capture_user_input(self):
        num_squares = input("Enter the number of squares the prawn is on: ")
        return num_squares
    
    def record_data(self, image_file_name, diameter_pixels, num_squares):

        return [image_file_name,diameter_pixels, convert_pixel_to_real_length(diameter_pixels), num_squares]

    def save_data_to_csv(self, data, filename='data.csv'):
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
    
    def run_viewer(self):
        import keyboard  # Make sure to import the keyboard library

        while True:
            self.show_image()

            # Wait for a key press
            while True:
                if keyboard.is_pressed('d'):
                    self.navigate_images('d')
                    break
                elif keyboard.is_pressed('a'):
                    self.navigate_images('a')
                    break
                elif keyboard.is_pressed('v'):
                    self.navigate_segmentations('v')
                    break
                elif keyboard.is_pressed('z'):
                    self.navigate_segmentations('z')
                    break
                elif keyboard.is_pressed('r'):
                    num_squares = self.capture_user_input()
                    image_file_name = self.images[self.current_image_index]
                    # Example diameter value, replace with actual calculation
              # Replace with actual diameter calculation
                    data = self.record_data(image_file_name, self.length_by_calc, num_squares)
                    self.save_data_to_csv(data)
                    print("Data recorded.")
                elif keyboard.is_pressed('q'):
                    cv2.destroyAllWindows()  # Using 'q' to quit
                    break

            # 
            
