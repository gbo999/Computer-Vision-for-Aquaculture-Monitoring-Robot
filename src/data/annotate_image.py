import cv2

def annotate_image(image_path, label_data):
    """
    Annotates an image with bounding boxes and keypoints based on label data.

    Args:
        image_path (str): The path to the image file.
        label_data (list): A list of strings representing the label data. Each string is a line from the label file.

    Returns:
        None
    """
    # Load your image
    image = cv2.imread(image_path)

    # Image dimensions
    height, width, _ = image.shape

    for data in label_data:
        parts = data.split()
        # Extract bounding box coordinates
        x_center, y_center, bbox_width, bbox_height = [float(val) for val in parts[1:5]]
        x_center, y_center, bbox_width, bbox_height = int(x_center * width), int(y_center * height), int(bbox_width * width), int(bbox_height * height)
        top_left = (x_center - bbox_width // 2, y_center - bbox_height // 2)
        bottom_right = (x_center + bbox_width // 2, y_center + bbox_height // 2)

        # Draw bounding box
        cv2.rectangle(image, top_left, bottom_right, (255,0,0), 2)

        # Draw keypoints
        num_keypoints = (len(parts) - 5) // 3
        for i in range(num_keypoints):
            px, py = [float(val) for val in parts[5 + i*3: 7 + i*3]]  # Extract x, y coordinates
            px, py = int(px * width), int(py * height)  # Scale coordinates
            visibility = int(parts[7 + i*3])  # Extract visibility
            if visibility == 2:  # If the keypoint is visible
                cv2.circle(image, (px, py), 5, (0, 255, 0), -1)  # Draw the keypoint
                cv2.putText(image, str(i), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label the keypoint with its index

    # Display the result
    cv2.imshow('Annotated Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "/path/to/your/image.jpg"  # Replace with the path to your image file
    label_data = ["label line 1", "label line 2", "label line 3"]  # Replace with your label data

    annotate_image(image_path, label_data)    
