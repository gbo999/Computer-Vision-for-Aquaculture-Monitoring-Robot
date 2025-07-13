import pandas as pd
import fiftyone as fo

# Load the CSV file
csv_path = "spreadsheet_files/manual_classifications_with_bboxes.csv"
df = pd.read_csv(csv_path)

print("Analyzing CSV file for duplicate classifications...")

# Group by image name and manual_size to find duplicates
duplicates = df.groupby(['image_name', 'manual_size']).size().reset_index(name='count')
duplicates = duplicates[duplicates['count'] > 1]

if len(duplicates) > 0:
    print("\nFound duplicate classifications:")
    print("-" * 50)
    for _, row in duplicates.iterrows():
        print(f"\nImage: {row['image_name']}")
        print(f"Size: {row['manual_size']}")
        print(f"Count: {row['count']}")
        
        # Get all entries for this image/size combination
        entries = df[(df['image_name'] == row['image_name']) & 
                    (df['manual_size'] == row['manual_size'])]
        print("\nBounding box details:")
        for _, entry in entries.iterrows():
            print(f"- Box: x={entry['bbox_x']:.3f}, y={entry['bbox_y']:.3f}, "
                  f"w={entry['bbox_width']:.3f}, h={entry['bbox_height']:.3f}")
    
    # Try to load and tag the dataset
    try:
        dataset = fo.load_dataset("prawn_keypoints")
        print("\nTagging samples in FiftyOne dataset...")
        
        for _, row in duplicates.iterrows():
            # Find the sample by matching the base filename
            base_name = row['image_name'].replace('colored_', '')
            for sample in dataset:
                if base_name in sample.filepath:
                    tag = f"duplicate_{row['manual_size']}"
                    if tag not in sample.tags:
                        sample.tags.append(tag)
                        print(f"Tagged {base_name} with '{tag}'")
                    break
        
        dataset.save()
        print("\nYou can view the tagged samples in FiftyOne with:")
        print("dataset.match_tags('duplicate_big').app()")
        print("or")
        print("dataset.match_tags('duplicate_small').app()")
        
    except Exception as e:
        print(f"\nCouldn't tag FiftyOne dataset: {e}")
        
else:
    print("No duplicate classifications found in the CSV file.") 