import os
import fiftyone as fo

# Path to the exported dataset - use relative path from project root
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 3 levels to reach the project root, then to exported_datasets
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
EXPORTED_DATASET_DIR = os.path.join(project_root, "exported_datasets/carapace_all")

print(f"Attempting to load dataset from {EXPORTED_DATASET_DIR}...")

# Delete existing dataset if it exists
if fo.dataset_exists("carapace_all"):
    print("Deleting existing dataset...")
    fo.delete_dataset("carapace_all")

# Try to load the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=EXPORTED_DATASET_DIR,
    dataset_type=fo.types.FiftyOneDataset,
    name="carapace_all"
)

print(f"\nLoaded dataset with {len(dataset)} samples")
print("\nDataset fields:")
print(dataset.get_field_schema())

# Launch the app to view the dataset
print("\nLaunching FiftyOne app...")
session = fo.launch_app(dataset)
session.wait() 