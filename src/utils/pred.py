# predict_script.py

from ultralytics import YOLO
import os
from pathlib import Path

def run_predictions(
    model_path,
    source_dirs,
    img_size=(640, 360),
    conf=0.5,
    iou=0.5,
    project="runs/predict",
    name="exp"
):
    """
    Run YOLO predictions on multiple directories.
    """
    # Load model
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    # Prediction parameters
    pred_args = {
        'save': True,
        'save_txt': True,
        'show_boxes': True,
        'imgsz': img_size,
        'conf': conf,
        'iou': iou,
        'project': project,
        'name': name,
        'exist_ok': True
    }

    # Process each directory
    for source_dir in source_dirs:
        print(f"\nProcessing directory: {source_dir}")
        
        # Create output directory path
        output_dir = Path(project) / name / Path(source_dir).name
        print(f"Results will be saved to: {output_dir}")

        # Run predictions
        try:
            results = model.predict(
                source=source_dir,
                **pred_args
            )
            print(f"Successfully processed {source_dir}")
        except Exception as e:
            print(f"Error processing {source_dir}: {str(e)}")

if __name__ == "__main__":
    # Model path
    model_path = "/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/thesisi/best-all.pt"

    # Source directories to process
    source_dirs = [
       
        "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/image processed/good/640360/square"
    ]

    # Run predictions
    run_predictions(
        model_path=model_path,
        source_dirs=source_dirs,
        img_size=(640, 360),
        conf=0.1,
        iou=0.0001,
        project="runs/predict",
        name="keypoint_detection"
    )