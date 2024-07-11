import os
from tqdm import tqdm

def extract_bounding_boxes(src_folder, dst_folder):
    # Ensure the destination folder exists
    os.makedirs(dst_folder, exist_ok=True)

    # Process each text file in the source folder
    for filename in tqdm(os.listdir(src_folder)):
        if filename.endswith('.txt'):
            with open(os.path.join(src_folder, filename), 'r') as file:
                lines = file.readlines()
            
            # Create the output for bounding boxes only
            bounding_boxes = []
            for line in lines:
                parts = line.strip().split()
                class_index = parts[0]
                x_center = parts[1]
                y_center = parts[2]
                width = parts[3]
                height = parts[4]
                
                bounding_box = f"{class_index} {x_center} {y_center} {width} {height}\n"
                bounding_boxes.append(bounding_box)
            
            # Write the bounding boxes to a new file in the destination folder
            with open(os.path.join(dst_folder, filename), 'w') as file:
                file.writelines(bounding_boxes)

# Define the source and destination folders
src_folder = 'C:/Users/gbo10/OneDrive/measurement_paper_images/to colab/31-12-76/amphibina car.v16i.yolov8/valid/labels-keypoints'
dst_folder =  'C:/Users/gbo10/OneDrive/measurement_paper_images/to colab/31-12-76/amphibina car.v16i.yolov8/valid/labels-bounding-boxes'

# Run the extraction process
extract_bounding_boxes(src_folder, dst_folder)
