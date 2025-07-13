import fiftyone as fo

# Load the dataset
dataset = fo.load_dataset("prawn_keypoints")
print(f"Loaded dataset with {len(dataset)} samples")

# Process each sample
for sample in dataset:
    if not hasattr(sample, "detections") or sample.detections is None:
        continue
        
    # Count classifications for this sample
    big_count = 0
    small_count = 0
    
    for det in sample.detections.detections:
        if hasattr(det, "tags"):
            if 'big' in det.tags:
                big_count += 1
            elif 'small' in det.tags:
                small_count += 1
    
    # Tag samples with duplicate classifications
    if big_count > 1:
        sample.tags.append("multiple_big")
    if small_count > 1:
        sample.tags.append("multiple_small")
        
    # Print info for samples with duplicates
    if big_count > 1 or small_count > 1:
        print(f"\nImage: {sample.filepath}")
        print(f"Big detections: {big_count}")
        print(f"Small detections: {small_count}")

# Save the dataset
dataset.save()

print("\nDone! Added tags:")
print("- 'multiple_big': samples with multiple 'big' classifications")
print("- 'multiple_small': samples with multiple 'small' classifications")
print("\nYou can view these in the FiftyOne UI with:")
print("dataset.match_tags('multiple_big').app()")
print("or")
print("dataset.match_tags('multiple_small').app()") 