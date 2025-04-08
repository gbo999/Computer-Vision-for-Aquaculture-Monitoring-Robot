#!/usr/bin/env python3
"""
FiftyOne Remote Server for Prawn Measurements Dataset

This script sets up a FiftyOne server that allows remote connections
to view the prawn measurements dataset.
"""

import argparse
import fiftyone as fo
import os
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='FiftyOne Remote Server for Prawn Measurements')
    parser.add_argument('--type', choices=['carapace', 'body'], default='body',
                      help='Type of measurement to analyze')
    parser.add_argument('--weights_type', choices=['car', 'kalkar', 'all'], default='all',
                      help='Version of the prediction to use')
    parser.add_argument('--address', type=str, default='0.0.0.0',
                      help='Address to bind the server to')
    parser.add_argument('--port', type=int, default=5151,
                      help='Port for FiftyOne visualization')
    parser.add_argument('--remote_port', type=int, default=5252,
                      help='Port for FiftyOne remote server connection')
    return parser.parse_args()

def start_remote_server(measurement_type, weights_type, address, port, remote_port):
    """
    Start a FiftyOne remote server with the specified dataset
    """
    # Configure the dataset name
    dataset_name = f"prawn_dataset_{measurement_type}_{weights_type}"
    
    # Check if the dataset already exists
    if dataset_name in fo.list_datasets():
        print(f"Loading existing dataset: {dataset_name}")
        dataset = fo.load_dataset(dataset_name)
    else:
        # Check if there's an exported dataset we can import
        export_path = Path(f"/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il/thesisi/thesis document/{measurement_type}_{weights_type}")
        if export_path.exists():
            print(f"Importing dataset from: {export_path}")
            dataset = fo.Dataset.from_dir(
                dataset_dir=str(export_path),
                dataset_type=fo.types.FiftyOneDataset,
                name=dataset_name
            )
        else:
            # If we need to create the dataset from scratch
            from data_loader import load_data, create_dataset, process_images, load_data_body, create_dataset_body
            
            if measurement_type == 'carapace':
                filtered_data_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/src/measurement/ImageJ/Filtered_Data.csv'
                load_data_fn = load_data
                create_dataset_fn = create_dataset
            else:  # body
                filtered_data_path = '/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/src/measurement/ImageJ/final_full_statistics_with_prawn_ids_and_uncertainty - Copy.xlsx'
                load_data_fn = load_data_body
                create_dataset_fn = create_dataset_body

            metadata_path = "/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/data/test images.xlsx"
            
            # Load data and create dataset
            filtered_df, metadata_df = load_data_fn(filtered_data_path, metadata_path)
            dataset, _ = create_dataset_fn(measurement_type, weights_type)
            
            # Process the dataset (simplified)
            print("Creating dataset from scratch - this may take some time...")
            
            # Make dataset persistent
            dataset.persistent = True
            dataset.save()
    
    # Start the remote server
    print(f"\n{'='*60}")
    print(f"STARTING FIFTYONE REMOTE SERVER")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Server address: {address}")
    print(f"App port: {port}")
    print(f"Remote port: {remote_port}")
    print(f"\nConnection string for clients:")
    print(f"fiftyone app connect --destination {address}:{remote_port} --port {port}")
    print(f"\nPress Ctrl+C to shut down the server.")
    print(f"{'='*60}\n")
    
    # Launch the server
    session = fo.launch_app(dataset, port=port, address=address, remote=True, remote_port=remote_port)
    
    try:
        # Keep the session alive until interrupted
        session.wait()
    except KeyboardInterrupt:
        print("\nShutting down FiftyOne remote server...")
    finally:
        if 'session' in locals() and session:
            session.close()

if __name__ == "__main__":
    args = parse_args()
    start_remote_server(args.type, args.weights_type, args.address, args.port, args.remote_port) 