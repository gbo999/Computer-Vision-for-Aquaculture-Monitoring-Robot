#!/usr/bin/env python3

import argparse
import fiftyone as fo
import os
import pandas as pd
import sys
from importlib import reload
from data_loader_unified import (
    create_unified_dataset,
    process_unified_images
)
import socket
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Unified prawn measurement validation using FiftyOne')
    parser.add_argument('--weights_type', choices=['car', 'kalkar', 'all'], default='all',
                      help='Version of the prediction to use')
    parser.add_argument('--port', type=int, default=5159,
                      help='Port to use for the FiftyOne app')
    return parser.parse_args()

def get_paths(weights_type):
    """Get paths for images, predictions, and ground truth based on weights type."""
    images_paths = {
        'right': '/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace/right',
        'left': '/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace/left',
        'car': '/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/measurements/carapace/car'
    }

    if weights_type == 'car':
        predict_version = 'predict88'
    elif weights_type == 'kalkar':
        predict_version = 'predict90'
    else:
        predict_version = 'predict89'

    prediction_base = f"/Users/gilbenor/Documents/code_projects/msc/counting_research_algorithms/training and val output/runs/pose/{predict_version}/labels"
    ground_truth_base = "/Users/gilbenor/Downloads/Giant freshwater prawn carapace keypoint detection.v91i.yolov8/all/labels"
    
    paths = {
        'folders': list(images_paths.values()),
        'predictions': [prediction_base] * 3,
        'ground_truths': [ground_truth_base] * 3
    }
    
    return paths

def process_measurements(weights_type, port):
    """Process both carapace and body measurements into a unified dataset."""
    
    # Define data paths
    base_path = '/Users/gilbenor/Documents/code_projects/msc/counting_research_algorithms'
    filtered_data_path = os.path.join(base_path, 'fifty_one and analysis/measurements/imagej/spreadsheet_files')
    
    # Load data files
    carapace_path = os.path.join(filtered_data_path, 'Filtered_Data.csv')
    body_path = os.path.join(filtered_data_path, 'final_full_statistics_with_prawn_ids_and_uncertainty - Copy.xlsx')
    metadata_path = os.path.join(filtered_data_path, 'test images.xlsx')
    
    # Load metadata and measurement data
    metadata_df = pd.read_excel(metadata_path)
    carapace_df = pd.read_csv(carapace_path)
    body_df = pd.read_excel(body_path)
    
    # Create or load dataset
    dataset, exists = create_unified_dataset(weights_type)
    
    if exists:
        print(f"Loading existing dataset: {dataset.name}")
        dataset.load_saved_view()
    else:
        print(f"Creating new dataset: {dataset.name}")
        
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
            
            # Extract pond type from path
            pond_tag = folder_path.split('/')[-1]
            print(f"Processing {pond_tag}")
            
            # Process images
            carapace_df, body_df = process_unified_images(
                image_paths=image_paths,
                prediction_folder_path=prediction_folder_path,
                ground_truth_paths_text=ground_truth_paths,
                carapace_df=carapace_df,
                body_df=body_df,
                metadata_df=metadata_df,
                dataset=dataset,
                pond_type=pond_tag
            )
        
        # Save the dataset
        dataset.save()
        print("Dataset saved successfully")
        
        # Save updated measurement data
        carapace_output = f'updated_filtered_data_with_lengths_carapace-{weights_type}.xlsx'
        body_output = f'updated_filtered_data_with_lengths_body-{weights_type}.xlsx'
        carapace_df.to_excel(carapace_output, index=False)
        body_df.to_excel(body_output, index=False)
        print(f"Results saved to {carapace_output} and {body_output}")
    
    # Launch the app
    session = fo.launch_app(dataset, port=port)
    print(f"Dataset visualization available at http://localhost:{port}")
    
    # Keep the script running
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\nClosing FiftyOne App...")
        session.close()

def main():
    args = parse_args()
    process_measurements(args.weights_type, args.port)

if __name__ == "__main__":
    main() 