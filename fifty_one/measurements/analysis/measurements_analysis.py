#!/usr/bin/env python3

import argparse
import fiftyone as fo
import os
import pandas as pd
import sys
from importlib import reload
from data_loader import load_data, create_dataset, process_images, load_data_body, create_dataset_body

def parse_args():
    parser = argparse.ArgumentParser(description='Prawn measurement validation using FiftyOne')
    parser.add_argument('--type', choices=['carapace', 'body'],default='body',
                      help='Type of measurement to analyze')
    parser.add_argument('--weights_type', choices=['car', 'kalkar', 'all'], default='all',
                      help='Version of the prediction to use')
    parser.add_argument('--port', type=int, default=5153,
                      help='Port for FiftyOne visualization')
    return parser.parse_args()

def get_paths(weights_type):
    images_paths = {
        'right': '/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace_resized/right',
        'left': '/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace_resized/left',
        'car': '/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace_resized/car'
    }
    


    if weights_type == 'car':
        predict_version = 'predict52'
    elif weights_type == 'kalkar':
        predict_version = 'predict53'
    else:
        predict_version = 'predict54'

    prediction_base = f"/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/runs/pose/{predict_version}/labels"
    ground_truth_base = "/Users/gilbenor/Downloads/Giant freshwater prawn carapace keypoint detection.v91i.yolov8/all/labels"
    
    paths = {
        'folders': list(images_paths.values()),
        'predictions': [prediction_base] * 3,
        'ground_truths': [ground_truth_base] * 3
    }
    
    return paths

def process_measurements(measurement_type, port, weights_type):
    # Define data file paths based on measurement type
    if measurement_type == 'carapace':
        filtered_data_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/src/measurement/ImageJ/Filtered_Data.csv'
        output_file = f'updated_filtered_data_with_lengths_carapace-{weights_type}.xlsx'
        keypoint_classes = ["start-carapace", "eyes"]
        load_data_fn = load_data
        create_dataset_fn = create_dataset
    else:  # body
        filtered_data_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/src/measurement/ImageJ/final_full_statistics_with_prawn_ids_and_uncertainty - Copy.xlsx'
        output_file = f'updated_filtered_data_with_lengths_body-{weights_type}.xlsx'
        keypoint_classes = ["tail", "rostrum"]
        load_data_fn = load_data_body
        create_dataset_fn = create_dataset_body

    metadata_path = "/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/test images.xlsx"
    
    # Load data and create dataset
    filtered_df, metadata_df = load_data_fn(filtered_data_path, metadata_path)
    dataset = create_dataset_fn()
    
    # Get paths for processing
    paths = get_paths(weights_type)
    
    # Process each test set
    for folder_path, prediction_folder_path, ground_truth_folder_path in zip(
        paths['folders'], paths['predictions'], paths['ground_truths']):
        
        # Get file paths
        image_paths = [
            os.path.join(folder_path, img) 
            for img in os.listdir(folder_path) 
            if img.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        ground_truth_paths = [
            os.path.join(ground_truth_folder_path, txt) 
            for txt in os.listdir(ground_truth_folder_path) 
            if txt.endswith('.txt')
        ]
        
        prediction_paths = [
            os.path.join(prediction_folder_path, txt) 
            for txt in os.listdir(prediction_folder_path) 
            if txt.endswith('.txt')
        ]
        
        # Extract pond type from path
        pond_tag = folder_path.split('/')[-1]
        print(f"Processing {pond_tag}")
        
        # Process images
        filtered_df = process_images(
            measurement_type=measurement_type,
            
            image_paths=image_paths,
            ground_truth_paths_text=ground_truth_paths,
            prediction_folder_path=prediction_folder_path,
            filtered_df=filtered_df,
            metadata_df=metadata_df,
            dataset=dataset,
            pond_type=pond_tag
        )


    # Evaluate keypoint detection performance
    results = dataset.evaluate_detections(
        "keypoints",
        gt_field="keypoints_truth",
        eval_key="pose_eval",
        method="coco",
        compute_mAP=True,
        iou=0.5,
        use_keypoints=True,
        classes=keypoint_classes
    )

    # Add IoU scores to filtered_df
    for sample in dataset:
        for keypoint in sample.keypoints.keypoints:
            if hasattr(keypoint, 'pose_eval_iou'):
                filtered_df.loc[
                    (filtered_df['id'] == keypoint.id), 
                    'pose_eval_iou'
                ] = keypoint.pose_eval_iou

    # Save results
    filtered_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Launch FiftyOne UI for visualization
    session = fo.launch_app(dataset, port=port)
    return session

def main():
    args = parse_args()
    session = process_measurements(args.type, args.port, args.weights_type)
    
    # Keep the session alive until user interrupts
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nClosing FiftyOne session...")
        session.close()

if __name__ == "__main__":
    main() 