import pandas as pd
import fiftyone as fo

# Load the CSV file
df = pd.read_csv("spreadsheet_files/manual_classifications_with_bboxes.csv")

# Find duplicate images
duplicates = df.groupby(['image_name', 'manual_size']).size().reset_index(name='count')
duplicate_images = duplicates[duplicates['count'] > 1]['image_name'].unique()

print(f"Found {len(duplicate_images)} images with duplicates:")
for img in duplicate_images:
    print(f"- {img}")

# Load dataset and tag samples
dataset = fo.load_dataset("prawn_keypoints")

for sample in dataset:
    for img_name in duplicate_images:
        base_name = img_name.replace('colored_', '')
        if base_name in sample.filepath:
            if "has_duplicates" not in sample.tags:
                sample.tags.append("has_duplicates")
                sample.save()
                print(f"Tagged: {base_name}")
            break

dataset.save()

# Launch FiftyOne with duplicates view
duplicates_view = dataset.match_tags("has_duplicates")
session = fo.launch_app(duplicates_view)
session.wait() 