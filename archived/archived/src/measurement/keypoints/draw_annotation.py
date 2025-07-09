import cv2
import numpy as np

def draw_keypoints_and_lines(image_path, labels_path):
    # Load the image
    image = cv2.imread(image_path)

    # Read the keypoints from the labels file
    keypoints = []
    with open(labels_path, 'r') as file:
        for line in file:
            x, y = map(int, line.strip().split())
            keypoints.append((x, y))

    # Draw keypoints on the image
    for point in keypoints:
        cv2.circle(image, point, 5, (0, 0, 255), -1)

    # Draw lines between keypoints
    for i in range(len(keypoints) - 1):
        cv2.line(image, keypoints[i], keypoints[i+1], (0, 255, 0), 2)

    # Calculate and display Euclidean distances
    for i in range(len(keypoints) - 1):
        distance = np.linalg.norm(np.array(keypoints[i]) - np.array(keypoints[i+1]))
        cv2.putText(image, f'{distance:.2f}', keypoints[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'path/to/source/image.jpg'
labels_path = 'path/to/keypoints/labels.txt'
draw_keypoints_and_lines(image_path, labels_path)