import cv2
import os
import numpy as np

def resize_image(image_path, target_size=(640, 640)):
    """
    Resize an image to the specified target size.

    Args:
        image_path (str): The path to the image file.
        target_size (tuple, optional): The target size to resize the image to. Defaults to (640, 640).

    Returns:
        numpy.ndarray: The resized image.

    """
    image = cv2.imread(image_path)
    return cv2.resize(image, target_size)

def adjust_labels(label_path, orig_size, target_size):
    """
    Adjusts the coordinates in a label file based on the original size and target size.

    Args:
        label_path (str): The path to the label file.
        orig_size (tuple): The original size of the image (width, height).
        target_size (tuple): The target size of the image (width, height).

    Returns:
        list: A list of strings representing the adjusted lines in the label file.
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        coords = list(map(float, parts[:8]))  # Extracting the coordinates
        class_name, difficulty = parts[8], parts[9]

        # Scale coordinates
        scaled_coords = []
        for i in range(0, len(coords), 2):  # Processing x, y pairs
            x_scaled = coords[i] * target_size[0] / orig_size[0]
            y_scaled = coords[i + 1] * target_size[1] / orig_size[1]
            scaled_coords.extend([x_scaled, y_scaled])

        new_line = ' '.join(map(str, scaled_coords)) + f" {class_name} {difficulty}\n"
        new_lines.append(new_line)

    return new_lines

def process_images_and_labels(images_folder, labels_folder, output_images_folder, output_labels_folder, target_size=(640, 640)):
    """
    Process images and labels by resizing the images and adjusting the labels.

    Args:
        images_folder (str): Path to the folder containing the input images.
        labels_folder (str): Path to the folder containing the input labels.
        output_images_folder (str): Path to the folder where the resized images will be saved.
        output_labels_folder (str): Path to the folder where the adjusted labels will be saved.
        target_size (tuple, optional): The target size to resize the images to. Defaults to (640, 640).

    Returns:
        None
    """
    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)
        label_path = os.path.join(labels_folder, os.path.splitext(image_name)[0] + '.txt')

        resized_image = resize_image(image_path, target_size)
        adjusted_labels = adjust_labels(label_path, (640, 360), target_size)

        # Save resized image
        cv2.imwrite(os.path.join(output_images_folder, image_name), resized_image)

        # Save adjusted labels
        with open(os.path.join(output_labels_folder, os.path.splitext(image_name)[0] + '.txt'), 'w') as file:
            file.writelines(adjusted_labels)

# Example usage
process_images_and_labels('C:/Users/gbo10/Videos/research/counting_research_algorithms/src/false/train/images', 'C:/Users/gbo10/Videos/research/counting_research_algorithms/src/false/train/labelTxt', "C:/Users/gbo10/Videos/research/counting_research_algorithms/src/true/train/images", "C:/Users/gbo10/Videos/research/counting_research_algorithms/src/true/train/labelTxt")
