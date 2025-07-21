import os
import shutil
from pathlib import Path

def organize_test_data(source_dir, destination_base_dir):
    """
    Organize test images and labels based on right/left/car folder contents.
    
    Args:
        source_dir (str): Path to source directory containing test images and labels
        destination_base_dir (str): Base path for organized folders
    """
    # Create destination directories
    pond_types = ['right', 'left', 'car']
    for pond_type in pond_types:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(destination_base_dir, f'test-{pond_type}', subdir), exist_ok=True)

    # Create mapping of filenames to pond types
    filename_to_pond = {}
    for pond_type in pond_types:
        # Look in images folder for each pond type
        images_dir = os.path.join(source_dir, pond_type)
        if os.path.exists(images_dir):
            # Get base filenames without extensions
            filenames = [os.path.splitext(f)[0] for f in os.listdir(images_dir)]
            for fname in filenames:
                filename_to_pond[fname] = pond_type

    # Get all test files from both images and labels directories
    test_images_dir = os.path.join(source_dir, 'test', 'images')
    test_labels_dir = os.path.join(source_dir, 'test', 'labels')
    
    # Process images
    for filename in os.listdir(test_images_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(filename)[0]
            pond_type = filename_to_pond.get(base_name)
            
            if pond_type:
                src_path = os.path.join(test_images_dir, filename)
                dst_path = os.path.join(destination_base_dir, f'test-{pond_type}', 'images', filename)
                shutil.copy2(src_path, dst_path)
                print(f"Copied image {filename} to {dst_path}")
            else:
                print(f"Warning: Could not determine pond type for image {filename}")

    # Process labels
    for filename in os.listdir(test_labels_dir):
        if filename.endswith('.txt'):
            base_name = os.path.splitext(filename)[0]
            pond_type = filename_to_pond.get(base_name)
            
            if pond_type:
                src_path = os.path.join(test_labels_dir, filename)
                dst_path = os.path.join(destination_base_dir, f'test-{pond_type}', 'labels', filename)
                shutil.copy2(src_path, dst_path)
                print(f"Copied label {filename} to {dst_path}")
            else:
                print(f"Warning: Could not determine pond type for label {filename}")

if __name__ == "__main__":
    # Source directory containing the dataset
    source_dir = "/Users/gilbenor/Downloads/Giant freshwater prawn carapace keypoint detection.v91i.yolov8"
    
    # Destination directory for organized files
    destination_base_dir = "/Users/gilbenor/Downloads/organized_test_data"
    
    organize_test_data(source_dir, destination_base_dir) 