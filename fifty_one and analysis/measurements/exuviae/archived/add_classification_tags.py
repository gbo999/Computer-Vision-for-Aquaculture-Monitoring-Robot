import fiftyone as fo

# Load the dataset
dataset = fo.load_dataset("prawn_keypoints")
print(f"Loaded dataset with {len(dataset)} samples")

# Add classification tags to the dataset
dataset.default_classes = ["big", "small"]
dataset.default_mask_targets = ["big", "small"]

# Save the dataset
dataset.save()

print("\nAdded classification tags 'big' and 'small' to the dataset")
print("\nYou can now use the FiftyOne UI to:")
print("1. Select a detection")
print("2. Press 'b' to tag as 'big'")
print("3. Press 's' to tag as 'small'")
print("4. Press 'c' to clear tags")
print("\nLaunch the app with:")
print("dataset.app()") 