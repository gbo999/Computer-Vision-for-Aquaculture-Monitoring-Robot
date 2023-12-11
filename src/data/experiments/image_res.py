import cv2

# Load the image using OpenCV
image_path = 'C:/Users/gbo10/Videos/research/counting_research_algorithms/src/data/experiments/1.jpegW'  # Update this with your actual image path
image = cv2.imread(image_path)

# Get image resolution
height, width = image.shape[:2]

print(f"The image resolution is: {width}x{height} pixels")
