import fiftyone as fo

# Load the dataset
dataset = fo.load_dataset("prawn_keypoints")
print(f"Loaded dataset with {len(dataset)} samples")

# Create a view of samples with duplicate_big tag
duplicates_view = dataset.match_tags("duplicate_big")
print(f"\nFound {len(duplicates_view)} samples with duplicate 'big' classifications")

# Launch the app with the duplicates view
session = fo.launch_app(duplicates_view)

print("\nInstructions:")
print("1. In the FiftyOne UI, you'll see only the images with duplicate 'big' classifications")
print("2. For each detection in these images:")
print("   - Click on the detection to select it")
print("   - Press 'b' to tag as 'big'")
print("   - Press 's' to tag as 'small'")
print("   - Press 'c' to clear tags")
print("\nPress Ctrl+C in this terminal when you're done")

# Wait for the session
session.wait() 