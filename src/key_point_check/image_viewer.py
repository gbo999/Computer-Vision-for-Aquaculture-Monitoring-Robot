import cv2
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from measurements_calculator import calculate_real_width
import keyboard  # This needs to be installed via pip
import matplotlib.patches as patches
from matplotlib.backend_tools import ToolBase, ToolToggleBase
import time

class ImageViewer:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = self._load_images()
        self.current_image_name = self.images[0] if self.images else None
        self.length_by_calc = 0
        self.current_segmentation_index = 0
        self.labels = []
        self.fig, self.ax = plt.subplots()  # One-time creation of the plot
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_event)

    def parse_label_file(self, label_file_path):
        objects = []
        with open(label_file_path, 'r') as file:
            for line in file:
                parts = line.split()
                obj_info = {
                    'class_index': int(parts[0]),
                    'bbox': {
                        'center_x': float(parts[1]),
                        'center_y': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    },
                    'keypoints': [
                        {'x': float(parts[5]), 'y': float(parts[6]), 'visibility': float(parts[7])},
                        {'x': float(parts[8]), 'y': float(parts[9]), 'visibility': float(parts[10])}
                    ]
                    # 'confidence': float(parts[11])
                }
                objects.append(obj_info)
        return objects
    import matplotlib.pyplot as plt
    import numpy as np

    def draw_keypoints_and_lines(self, ax, keypoints, image_width, image_height):
        # Convert normalized coordinates to pixel coordinates and include visibility
        pixel_points = [(int(kp['x'] * image_width), int(kp['y'] * image_height), kp['visibility']) for kp in keypoints]

        # Draw lines between visible points and calculate distances
        visible_points = [point for point in pixel_points if point[2] > 0.5]
        if len(visible_points) > 1:
            for i in range(len(visible_points) - 1):
                x1, y1, _ = visible_points[i]
                x2, y2, _ = visible_points[i+1]
                # Drawing line between pixel keypoints
                ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2)  # Green lines



                # Calculate the Euclidean distance in pixels between points
                distance_px = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
                real_distance = calculate_real_width(24.4, 625, distance_px, 0.00716844)  # Placeholder for real-world distance calculation
                print(f"Distance between ({x1}, {y1}) and ({x2}, {y2}): {distance_px} px, Real length: {real_distance}")

                # Displaying the real distance on the image
                midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
                ax.annotate(f"{real_distance:.2f} mm", xy=midpoint, textcoords='offset points',
                            xytext=(0, 10), ha='center', color='white',
                            bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.3))



    def _load_images(self):
        return [img for img in sorted(os.listdir(self.image_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def _load_labels(self, image_name):
        label_name = os.path.splitext(image_name)[0] + '.txt'
        return os.path.join(self.label_dir, label_name)



    # def _draw_annotations(self, image):
    #         fig, ax = plt.subplots()
    #         ax.imshow(image)
    #         label_path = self._load_labels(self.current_image_name)
    #         if os.path.exists(label_path):
    #             with open(label_path, 'r') as file:
    #                 lines = file.readlines()
    #                 self.labels = lines
    #                 if self.current_segmentation_index < len(lines):
    #                     line = lines[self.current_segmentation_index]
    #                     parts = line.split()
    #                     keypoints = [(float(parts[i]), float(parts[i + 1])) for i in range(5, len(parts), 2)]
    #                     pixel_points = [(x * image.shape[1], y * image.shape[0]) for x, y in keypoints]
    #                     for point in pixel_points:
    #                         ax.plot(point[0], point[1], 'ro')
    #                     if len(pixel_points) > 1:
    #                         poly = patches.Polygon(pixel_points, linewidth=1, edgecolor='g', facecolor='none')
    #                         ax.add_patch(poly)
    #         plt.show()

    
    def _convert_to_pixels(self, normalized_points, image_width, image_height):
        return [(int(x * image_width), int(y * image_height)) for x, y in normalized_points]

    def navigate_images(self, key):
        index = self.images.index(self.current_image_name)
        if key == 'd' and index < len(self.images) - 1:
            self.current_image_name = self.images[index + 1]
            self.current_segmentation_index = 0
            self.show_image()
        elif key == 'a' and index > 0:
            self.current_image_name = self.images[index - 1]
            self.current_segmentation_index = 0
            self.show_image()


    def navigate_segmentations(self, key):
        if key == 'z' and self.current_segmentation_index > 0:
            self.current_segmentation_index -= 1
        elif key == 'v' and self.current_segmentation_index < len(self.labels) - 1:
            self.current_segmentation_index += 1

    

    def show_image(self, key=None):
        # if key is not None:
        #     self.navigate_images(key)
        #     #REDRAW THE FIGURE
        #     self.fig.canvas.draw()
        #     plt.show()
    
        if self.current_image_name:
            image_path = os.path.join(self.image_dir, self.current_image_name)
        image = cv2.imread(image_path)
        if image is None:
            print("Failed to load image:", image_path)
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # Clear the current axes and draw the new image
        self.ax.clear()
        self.ax.imshow(image)
    
        label_path = self._load_labels(self.current_image_name)
        objects = self.parse_label_file(label_path)
        for obj in objects:
            self.draw_keypoints_and_lines(self.ax, obj['keypoints'], image.shape[1], image.shape[0])
        # draw box
            box=obj['bbox']    
            x = box['center_x'] * image.shape[1]
            y = box['center_y'] * image.shape[0]
            width = box['width'] * image.shape[1]
            height = box['height'] * image.shape[0]
            rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=2, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
        # Draw the annotations

        # for obj in objects:         


        # self._draw_annotations(image) 
        # Manual marking and distance calculation
        

        
            

    
        # Redraw the figure
        self.fig.canvas.draw()
        plt.show()


    def run_viewer(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_event)
        self.show_image()

    def m_pressed(self):
        points = []
    
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                points.append((event.xdata, event.ydata))
            self.ax.plot(event.xdata, event.ydata, 'ro')
            if len(points) == 2:
                dist = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
                self.ax.annotate(f"{dist:.2f}px", xy=(np.mean([p[0] for p in points]), np.mean([p[1] for p in points])), textcoords='offset points', arrowprops=dict(facecolor='yellow', shrink=0.05))
                print("Distance between points:", dist, "pixels")
                self.fig.canvas.draw()
    
        cid = self.fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    def on_key_event(self, event):
        print(f"{event.key} is pressed")
        if event.key == 'd':
            self.navigate_images('d')
        elif event.key == 'a':
            self.navigate_images('a')
        elif event.key == 'm':
            self.m_pressed()
        elif event.key == 'q':
            plt.close() 
    # Add a small delay to avoid excessive CPU usage

# # Example of using the class
# viewer = ImageViewer('C:/Users/gbo10/OneDrive/research/italy/torino/computer vision/good for test-curtain open/images', 'C:/Users/gbo10/OneDrive/research/italy/torino/computer vision/good for test-curtain open/labels')
# viewer.run_viewer()
