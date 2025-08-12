import fiftyone as fo

# Load the dataset
dataset = fo.load_dataset("prawn_keypoints")
print(f"Loaded dataset with {len(dataset)} samples")

# Create a view of samples with multiple_big tag
duplicates_view = dataset.match_tags("multiple_big")
print(f"\nFound {len(duplicates_view)} samples with multiple 'big' classifications")

# Print the filenames of affected images
print("\nAffected images:")
for sample in duplicates_view:
    print(f"- {sample.filepath}")

# Launch the app with the duplicates view
session = fo.launch_app(duplicates_view)
session.wait() 