from image_viewer import ImageViewer
from data_recorder import record_data, save_data_to_csv

def main():
    viewer = ImageViewer(image_dir="C:\\Users\\gbo10\\OneDrive\\pictures\\labeling\\65\\65\\test\\images", label_dir="C:\\Users\\gbo10\\Videos\\research\\counting_research_algorithms\\notebooks\\artifacts\\after_1.1-v0\\segment\\predict\\labels")
    viewer.run_viewer()

if __name__ == "__main__":
    main()
