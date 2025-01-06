import os
from PIL import Image
import shutil

def organize_carapace_images(source_base_path, target_base_path, target_size=(640, 360)):
    """
    Organize and resize carapace images into a new folder structure.
    
    Args:
        source_base_path (str): Path to source images (/measurements/carapace)
        target_base_path (str): Path to target folder (carapace_resized)
        target_size (tuple): Target image dimensions (width, height)
    """
    # Create target folders
    folders = ['car', 'left', 'right']
    for folder in folders:
        target_folder = os.path.join(target_base_path, folder)
        os.makedirs(target_folder, exist_ok=True)
        
        # Source folder path
        source_folder = os.path.join(source_base_path, folder)
        
        # Process each image in the source folder
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # Source and target paths
                source_path = os.path.join(source_folder, filename)
                target_path = os.path.join(target_folder, filename)
                
                # Open, resize, and save image
                with Image.open(source_path) as img:
                    # Resize image maintaining aspect ratio
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    resized_img.save(target_path)
                    print(f"Processed: {filename} -> {target_folder}")

if __name__ == "__main__":
    # Source and target paths
    source_path = "/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace"
    target_path = os.path.join(os.path.dirname(source_path), "carapace_resized")
    
    # Create and process images
    organize_carapace_images(source_path, target_path)
    print("\nProcessing complete!") 