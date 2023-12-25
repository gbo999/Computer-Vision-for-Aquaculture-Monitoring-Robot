import cv2
import os
import numpy as np

def draw_bounding_boxes(image_path, label_path):
    # Read the image
    image = cv2.imread(image_path)

    # Read the label file and draw each bounding box
    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        coords = list(map(float, parts[:8]))
        class_name = parts[8]

        # Assuming coords are in the order x1, y1, x2, y2, x3, y3, x4, y4
        pts = np.array([[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)], np.int32)
        pts = pts.reshape((-1, 1, 2))

        # Draw the bounding box
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Put the class name near the first point
        cv2.putText(image, class_name, (int(coords[0]), int(coords[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return image

def visualize_labels(image_folder, label_folder, output_folder):
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, os.path.splitext(image_name)[0] + '.txt')

        # Draw bounding boxes on the image
        image_with_boxes = draw_bounding_boxes(image_path, label_path)

        # Save the image with bounding boxes
        cv2.imwrite(os.path.join(output_folder, image_name), image_with_boxes)

# Example usage
visualize_labels('C:/Users/gbo10/Videos/research/counting_research_algorithms/src/true/valid/images', 'C:/Users/gbo10/Videos/research/counting_research_algorithms/src/true/valid/labelTxt', 'C:/Users/gbo10/Videos/research/counting_research_algorithms/src/true/viz')
