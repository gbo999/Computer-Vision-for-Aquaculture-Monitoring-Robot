import os
import fiftyone as fo

# Path to the exported dataset
EXPORTED_DATASET_DIR = "fiftyone_datasets/exuviae_keypoints"

print(f"Attempting to load dataset from {EXPORTED_DATASET_DIR}...")

# Delete existing dataset if it exists
if fo.dataset_exists("prawn_keypoints"):
    print("Deleting existing dataset...")
    fo.delete_dataset("prawn_keypoints")

# Try to load the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=EXPORTED_DATASET_DIR,
    dataset_type=fo.types.FiftyOneDataset,
    name="prawn_keypoints"
)

print(f"\nLoaded dataset with {len(dataset)} samples")
print("\nDataset fields:")
print(dataset.get_field_schema())

# Launch the app to view the dataset
print("\nLaunching FiftyOne app...")
session = fo.launch_app(dataset)
session.wait() 