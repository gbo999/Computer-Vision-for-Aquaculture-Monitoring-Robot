import os
import fiftyone as fo

# Path to the exported dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 2 levels to reach the project root, then to exported_datasets/prawn_counting
project_root = os.path.dirname(os.path.dirname(current_dir))
EXPORTED_DATASET_DIR = os.path.join(project_root, "exported_datasets/prawn_counting") 

print(f"Attempting to load dataset from {EXPORTED_DATASET_DIR}...")

# Delete existing dataset if it exists
if fo.dataset_exists("prawn_counting"):
    print("Deleting existing dataset...")
    fo.delete_dataset("prawn_counting")

# Try to load the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=EXPORTED_DATASET_DIR,
    dataset_type=fo.types.FiftyOneDataset,
    name="prawn_counting"
)

print(f"\nLoaded dataset with {len(dataset)} samples")
print("\nDataset fields:")
print(dataset.get_field_schema())

# Launch the app to view the dataset
print("\nLaunching FiftyOne app...")
session = fo.launch_app(dataset)
session.wait() 