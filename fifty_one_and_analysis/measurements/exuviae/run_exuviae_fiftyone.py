"""
This script is responsible for managing and visualizing a dataset of prawn keypoints using the FiftyOne library. 
It attempts to load an existing dataset from a specified directory, deletes any pre-existing dataset with the same 
name to ensure a fresh start, and then loads the dataset for analysis. The script also launches the FiftyOne app 
to provide a graphical interface for exploring the dataset.

Key Steps:
1. Define the path to the exported dataset directory.
2. Check for and delete any existing dataset named 'prawn_keypoints'.
3. Load the dataset from the specified directory.
4. Print the number of samples and the dataset's field schema.
5. Launch the FiftyOne app for dataset visualization.

Dependencies:
- fiftyone: For dataset management and visualization.
"""

import os
import fiftyone as fo

# Adjust the path to point to the exported_datasets directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels to reach the project root, then to exported_datasets
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
EXPORTED_DATASET_DIR = os.path.join(project_root, "exported_datasets/exuviae_keypoints")

print(f"Attempting to load dataset from {EXPORTED_DATASET_DIR}...")




# Delete existing dataset if it exists
if fo.dataset_exists("prawn_counting"):
    print("Deleting existing dataset...")
    fo.delete_dataset("prawn_counting")

# Try to load the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=EXPORTED_DATASET_DIR,
    dataset_type=fo.types.FiftyOneDataset,
    name='exuviae_keypoints'
)

print(f"\nLoaded dataset with {len(dataset)} samples")
print("\nDataset fields:")
print(dataset.get_field_schema())

# Launch the app to view the dataset
print("\nLaunching FiftyOne app...")
session = fo.launch_app(dataset, port=5173)
session.wait() 


