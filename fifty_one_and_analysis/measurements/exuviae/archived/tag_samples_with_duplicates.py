import fiftyone as fo

# Load the dataset
dataset = fo.load_dataset("prawn_keypoints")
print(f"Loaded dataset with {len(dataset)} samples")

# Process each sample
duplicate_count = 0
for sample in dataset:
    # Count "big" detections in this sample
    big_count = 0
    if hasattr(sample, "detections") and sample.detections is not None:
        for det in sample.detections.detections:
            if det.label == "big":
                big_count += 1
    
    # If more than one "big" detection, tag the sample
    if big_count > 1:
        sample.tags.append("has_duplicate_big")
        duplicate_count += 1
        print(f"Found {big_count} 'big' detections in {sample.filepath}")

# Save the dataset
dataset.save()

print(f"\nTagged {duplicate_count} samples with 'has_duplicate_big'")

# Create a view of just the samples with duplicates
duplicates_view = dataset.match_tags("has_duplicate_big")

# Launch the app with the duplicates view
session = fo.launch_app(duplicates_view)

print("\nInstructions:")
print("1. The FiftyOne UI will show only samples with multiple 'big' detections")
print("2. Review each sample and its detections")
print("\nPress Ctrl+C in this terminal when you're done") 