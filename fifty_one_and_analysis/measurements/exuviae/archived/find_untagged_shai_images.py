import pandas as pd
import fiftyone as fo
import os

def find_untagged_shai_images():
    """
    Find images from Shai's measurements that haven't been manually tagged yet
    and launch FiftyOne to help tag them.
    """
    print("Loading datasets...")
    
    # Load Shai's measurements
    shai_df = pd.read_csv("spreadsheet_files/Results-shai-exuviae.csv")
    
    # Load manual classifications
    manual_csv_path = "spreadsheet_files/manual_classifications_updated_20250713_113720.csv"
    if os.path.exists(manual_csv_path):
        manual_df = pd.read_csv(manual_csv_path)
        # Get images that have been manually classified (not untagged)
        manually_tagged_images = set(manual_df[manual_df['manual_size'] != 'untagged']['image_name'].unique())
        print(f"Found {len(manually_tagged_images)} manually tagged images")
    else:
        manually_tagged_images = set()
        print("No manual classifications file found")
    
    # Clean up Shai's image names to match the expected format
    shai_df['image_name'] = shai_df['Label'].str.replace('Shai - exuviae:', '')
    shai_df['image_name'] = shai_df['image_name'].str.replace('colored_', '')
    shai_df['image_name'] = shai_df['image_name'] + '.jpg'
    
    # Get unique images from Shai's measurements
    shai_images = set(shai_df['image_name'].unique())
    
    # Find images that are in Shai's measurements but not manually tagged
    untagged_shai_images = shai_images - manually_tagged_images
    
    print(f"Total images in Shai's measurements: {len(shai_images)}")
    print(f"Already manually tagged images: {len(manually_tagged_images)}")
    print(f"Untagged images from Shai's measurements: {len(untagged_shai_images)}")
    
    if len(untagged_shai_images) == 0:
        print("All images from Shai's measurements have been manually tagged!")
        return
    
    print("\nUntagged images from Shai's measurements:")
    for img in sorted(untagged_shai_images):
        # Count how many detections Shai made for this image
        count = len(shai_df[shai_df['image_name'] == img])
        print(f"- {img} ({count} detections)")
    
    # Load FiftyOne dataset
    try:
        dataset = fo.load_dataset("prawn_keypoints")
        print(f"\nLoaded FiftyOne dataset with {len(dataset)} samples")
        
        # Find samples that match the untagged images
        untagged_samples = []
        for sample in dataset:
            filename = os.path.basename(sample.filepath)
            if filename in untagged_shai_images:
                untagged_samples.append(sample)
        
        print(f"Found {len(untagged_samples)} samples in FiftyOne that need tagging")
        
        if len(untagged_samples) > 0:
            # Create a view with just the untagged samples
            sample_ids = [sample.id for sample in untagged_samples]
            untagged_view = dataset.select(sample_ids)
            
            # Tag these samples for easy identification
            for sample in untagged_samples:
                if "needs_manual_tagging" not in sample.tags:
                    sample.tags.append("needs_manual_tagging")
                    sample.save()
            
            dataset.save()
            
            print("\nTagged samples with 'needs_manual_tagging' for easy identification")
            print("\nLaunching FiftyOne with untagged samples...")
            print("\nInstructions for tagging:")
            print("1. Click on a detection to select it")
            print("2. Press 'b' to tag as 'big'")
            print("3. Press 's' to tag as 'small'")
            print("4. Press 'c' to clear tags")
            print("5. Use the arrow keys to navigate between samples")
            print("6. Press Ctrl+C in this terminal when done")
            
            # Launch FiftyOne with the untagged view
            session = fo.launch_app(untagged_view)
            session.wait()
        
    except Exception as e:
        print(f"Error loading FiftyOne dataset: {e}")
        print("Make sure the 'prawn_keypoints' dataset exists")

if __name__ == "__main__":
    find_untagged_shai_images() 