from image_viewer import ImageViewer
from data_recorder import record_data, save_data_to_csv

def main():
    viewer = ImageViewer(image_dir='path_to_images', label_dir='path_to_labels')
    viewer.run_viewer()

if __name__ == "__main__":
    main()
