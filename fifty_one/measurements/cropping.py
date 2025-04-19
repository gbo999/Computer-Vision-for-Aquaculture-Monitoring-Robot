import cv2
import os
from tqdm import tqdm
# --- CONFIG ---
images_folder = "/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace/contrast_stretched"            # folder with .jpg or .png
labels_folder = "/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/runs/pose/predict90/labels/"            # folder with YOLO .txt files
base_output_folder = "/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace/crop-num"
# Create 5 subdirectories
output_folders = []
for i in range(5):
    folder = os.path.join(base_output_folder, f"folder_{i+1}")
    output_folders.append(folder)
    os.makedirs(folder, exist_ok=True)

# Loop through all image files
for filename in tqdm(os.listdir(images_folder)):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    # Load image
    image_path = os.path.join(images_folder, filename)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Load corresponding label file
    txt_name = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(labels_folder, txt_name)
    if not os.path.exists(txt_path):
        print(f"No label file for {filename}, skipping.")
        continue

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        parts = line.strip().split()
        # if len(parts) != 5:
        #     continue

        class_id, x_center, y_center, box_w, box_h = map(float, parts[:5])

        # Convert from normalized to absolute pixel coordinates
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        # Crop the image using bounding box (make sure it stays within bounds)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Add 100 pixel padding around the bounding box (but stay within image boundaries)
        pad_x1 = max(0, x1 - 200)
        pad_y1 = max(0, y1 - 200)
        pad_x2 = min(w, x2 + 200)
        pad_y2 = min(h, y2 + 200)
        
        # Verify the padded crop is valid
        if pad_x2 <= pad_x1 or pad_y2 <= pad_y1:
            print(f"Skipping obj{idx} in {filename}: invalid padded dimensions")
            continue
            
        cropped = img[pad_y1:pad_y2, pad_x1:pad_x2]
        
        # Verify the cropped image is not empty
        if cropped.size == 0:
            print(f"Skipping obj{idx} in {filename}: empty crop")
            continue
        # Determine which output folder to use based on the object index
        folder_idx = idx % len(output_folders)
        output_folder = output_folders[folder_idx]

        # Save cropped image
        out_name1 = f"{os.path.splitext(filename)[0]}_obj{idx}_cropped{ x_center/w}_{y_center/h}-1.jpg"
        out_name2 = f"{os.path.splitext(filename)[0]}_obj{idx}_cropped{ x_center/w}_{y_center/h}-2.jpg"
        out_name3 = f"{os.path.splitext(filename)[0]}_obj{idx}_cropped{ x_center/w}_{y_center/h}-3.jpg"
        out_path1 = os.path.join(output_folder, out_name1)
        out_path2 = os.path.join(output_folder, out_name2)
        out_path3 = os.path.join(output_folder, out_name3)
        cv2.imwrite(out_path1, cropped)
        cv2.imwrite(out_path2, cropped)
        cv2.imwrite(out_path3, cropped)
