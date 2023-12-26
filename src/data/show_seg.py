import cv2
import numpy as np

def draw_polygon(img, class_id, points):
    # Convert normalized coordinates to pixel coordinates
    img_height, img_width = img.shape[:2]
    pts = np.array([[int(x * img_width), int(y * img_height)] for x, y in points], np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw the polygon
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(img, str(class_id), tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return pts

def draw_diameter_line(img, points):
    # Find the two farthest points in the segmentation
    max_distance = 0
    point1 = points[0][0]
    point2 = points[0][0]

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i][0]
            p2 = points[j][0]
            distance = np.linalg.norm(p1 - p2)
            
            if distance > max_distance:
                max_distance = distance
                point1 = p1
                point2 = p2

    # Draw the diameter line
    cv2.line(img, tuple(point1), tuple(point2), (0, 0, 255), 2)



def process_image(image_path, label_path):
    # Load the image and resize
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {image_path}")
    img = cv2.resize(img, (640, 360))

    # Read the YOLO label file
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.split()
            class_id = int(parts[0])
            points = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
            pts = draw_polygon(img, class_id, points)
            draw_diameter_line(img, pts)

    # Display the image
    cv2.imshow('Segmented Image with Diameters', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Example usage
process_image('C:/Users/gbo10/Videos/research/counting_research_algorithms/src/to_colab/valid/images/GX010063_MP4-37_jpg.rf.1e299b123582106ea7e5baa9dd3cc866.jpg'
, 'C:/Users/gbo10/Dropbox/research videos/21.12/seg/zipfile (2)/content/runs/segment/predict/labels/GX010063_MP4-37_jpg.rf.1e299b123582106ea7e5baa9dd3cc866.txt')
