import cv2
import os
import numpy as np
from src.measurement.segmentation import minimum_enclosing_circle
from measurements_calculator import convert_pixel_to_real_length
import csv

 
class ImageViewer:
    """
    A class for viewing the segmentation annotated images and .

    Args:
        image_dir (str): The directory path where the images are stored.
        label_dir (str): The directory path where the labels are stored.

    Attributes:
        image_dir (str): The directory path where the images are stored.
        label_dir (str): The directory path where the labels are stored.
        images (list): A list of image file names in the image directory.
        current_image_name (str): The file name of the current image being viewed.
        length_by_calc (float): The calculated length of an annotation.
        current_segmentation_index (int): The index of the current segmentation being viewed.
        labels (list): A list of label lines for the current image.

    Methods:
        _load_images: Load the image file names from the image directory.
        _load_labels: Load the label file for a given image.
        resize_image_to_screen: Resize an image to fit the screen dimensions.
        show_image: Show the current image with annotations.
        _draw_annotations: Draw annotations on the image.
        _draw_single_annotation: Draw a single annotation on the image.
        _convert_to_pixels: Convert normalized points to pixel coordinates.
        navigate_images: Navigate to the next or previous image.
        navigate_segmentations: Navigate to the next or previous segmentation.
        capture_user_input: Capture user input for the number of squares.
        record_data: Record data for the current image.
        save_data_to_csv: Save data to a CSV file.
        run_viewer: Run the image viewer loop.
    """
    
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.images = self._load_images()
        self.current_image_name = self.images[0] if self.images else None
        self.length_by_calc = 0
        self.current_segmentation_index = 0
        self.labels=[]

        
    def _load_images(self):
        """
        Load and return a list of images from the specified image directory.

        Returns:
            A list of image filenames, sorted in ascending order.

        """
        return [img for img in sorted(os.listdir(self.image_dir)) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def _load_labels(self, image_name):
            """
            Load the labels for the given image.

            Args:
                image_name (str): The name of the image.

            Returns:
                str: The path to the label file corresponding to the image.
            """
            label_name = os.path.splitext(image_name)[0] + '.txt'
            return os.path.join(self.label_dir, label_name)
   
    def resize_image_to_screen(self, image, screen_width, screen_height, keep_aspect_ratio=True):
        """
        Resize the given image to fit within the specified screen dimensions.

        Args:
            image (numpy.ndarray): The image to be resized.
            screen_width (int): The width of the screen.
            screen_height (int): The height of the screen.
            keep_aspect_ratio (bool, optional): Whether to maintain the aspect ratio of the image. 
                Defaults to True.

        Returns:
            numpy.ndarray: The resized image.
        """
        img_height, img_width = image.shape[:2]

        if keep_aspect_ratio:
            # Calculate the ratio of the screen dimensions to the image dimensions
            width_ratio = screen_width / img_width
            height_ratio = screen_height / img_height

            # Use the smaller ratio to ensure the entire image fits within the screen
            resize_ratio = min(width_ratio, height_ratio)
            new_width = int(img_width * resize_ratio)
            new_height = int(img_height * resize_ratio)

            image = cv2.resize(image, (new_width, new_height))
        else:
            image = cv2.resize(image, (screen_width, screen_height))

        return image

    def show_image(self):
        """
        Displays the current image in a window.

        This method loads the image specified by `self.current_image_name` from the image directory,
        resizes it to fit the screen, draws any annotations on the image, and displays it in a window.

        Returns:
            None
        """
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

    def _draw_annotations(self, image):
        """
        Draws annotations on the given image based on the loaded labels.

        Args:
            image (PIL.Image.Image): The image on which the annotations will be drawn.

        Returns:
            None
        """
        label_path = self._load_labels(self.current_image_name)
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
                self.labels = lines
                if self.current_segmentation_index < len(lines):
                    line = lines[self.current_segmentation_index]
                    parts = line.split()
                    points = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
                    self._draw_single_annotation(image, points)

    def _draw_single_annotation(self, image, points):
        """
        Draws a single annotation on the given image.

        Args:
            image (numpy.ndarray): The image on which to draw the annotation.
            points (list): The list of points representing the annotation.

        Returns:
            None
        """
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
            print(f"Length by calc: {self.length_by_calc}")
            text_pos2 = (int(center[0] ), int(center[1]+30))
            # Put the diameter text on the image
            cv2.putText(image, 
                        f"Diameter_real: {convert_pixel_to_real_length(self.length_by_calc)}", 
                        text_pos2, 
                        font, 
                        font_scale, 
                        font_color, 
                        line_type)
            print(f'diameter real: {convert_pixel_to_real_length(self.length_by_calc)}')

    def _convert_to_pixels(self, normalized_points, image_width, image_height):
        """
        Converts a list of normalized points to pixel coordinates based on the given image dimensions.

        Args:
            normalized_points (list): A list of tuples representing normalized points in the range [0, 1].
            image_width (int): The width of the image in pixels.
            image_height (int): The height of the image in pixels.

        Returns:
            list: A list of tuples representing the pixel coordinates of the normalized points.
        """
        pixel_points = []
        print(f'len {len(normalized_points)}')
        for x_normalized, y_normalized in normalized_points:
            x_pixel = int(x_normalized * image_width)
            y_pixel = int(y_normalized * image_height)
            pixel_points.append((x_pixel, y_pixel))
        return pixel_points

    def navigate_images(self, key):
        """
        Navigates through the images based on the provided key.

        Args:
            key (str): The key representing the navigation action. 'd' for moving to the next image,
                       'a' for moving to the previous image.

        Returns:
            None
        """
        current_index = self.images.index(self.current_image_name)
        if key == 'd' and current_index < len(self.images) - 1:
            self.current_image_name = self.images[current_index + 1]
            self.current_segmentation_index = 0  # Reset segmentation index
        elif key == 'a' and current_index > 0:
            self.current_image_name = self.images[current_index - 1]
            self.current_segmentation_index = 0  # Reset segmentation index

 
            

    def navigate_segmentations(self, key):
        """
        Navigates through the segmentations based on the provided key.

        Args:
            key (str): The key representing the navigation action. 
                       'z' to navigate to the previous segmentation.
                       'v' to navigate to the next segmentation.

        Returns:
            None
        """
        if key == 'z' and self.current_segmentation_index > 0:
            self.current_segmentation_index -= 1
            # Refresh the image with the new segmentation
        elif key == 'v' and self.current_segmentation_index < len(self.labels) - 1:
            self.current_segmentation_index += 1

    
    
    def record_data(self, image_file_name, diameter_pixels):
        """
        Records the data for an image file.

        Args:
            image_file_name (str): The name of the image file.
            diameter_pixels (float): The diameter of the object in pixels.

        Returns:
            list: A list containing the image file name, diameter in pixels, and the converted diameter in real length.
        """
        return [image_file_name, diameter_pixels, convert_pixel_to_real_length(diameter_pixels)]
    



    def save_data_to_csv(self, data, filename='data.csv'):
        """
        Saves the given data to a CSV file.

        Args:
            data (list): The data to be saved.
            filename (str, optional): The name of the CSV file. Defaults to 'data.csv'.
        """
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    
    def run_viewer(self):
        """
        Runs the image viewer and handles user interactions.

        This method continuously displays images and waits for user input to navigate through images,
        navigate through segmentations, record data, or quit the viewer.

        Keyboard shortcuts:
        - 'd': Navigate to the next image
        - 'a': Navigate to the previous image
        - 'v': Navigate to the next segmentation
        - 'z': Navigate to the previous segmentation
        - 'r': Record data for the current image
        - 'q': Quit the viewer

        Returns:
        None
        """
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
                    image_file_name = self.current_image_name
                    # Example diameter value, replace with actual calculation
                    # Replace with actual diameter calculation
                    data = self.record_data(image_file_name, self.length_by_calc)
                    self.save_data_to_csv(data)
                    print("Data recorded.")
                elif keyboard.is_pressed('q'):
                    cv2.destroyAllWindows()  # Using 'q' to quit
                    break

            # Continue displaying images

def main():
    
    viewer = ImageViewer(image_dir="C:/Users/gbo10/OneDrive/pictures/labeling/65/65/undistorted", label_dir="C:\\Users\\gbo10\\OneDrive\\pictures\\labeling\\65\\labels")
    viewer.run_viewer()

if __name__ == "__main__":
    main()

