"""
This script manages a FiftyOne dataset for exuviae keypoints with tag persistence and export.

What this script does differently compared to `run_exuviae_fiftyone.py`:
- It DOES NOT delete the dataset on each run. Instead, it loads an existing dataset if present.
  Quote: "if fo.dataset_exists(DATASET_NAME): ... dataset = fo.load_dataset(DATASET_NAME)"
- It exports sample-level tags to a CSV after you close the app.
  Quote: "writer.writerow([\"sample_id\", \"filepath\", \"tags\"])"
- It re-exports the dataset to the same directory so updated tags are saved with the dataset manifest.
  Quote: "dataset.export(... dataset_type=fo.types.FiftyOneDataset, overwrite=True)"

Usage:
- Run the script, add tags in the FiftyOne app, and close the app.
- The script writes `sample_tags.csv` to the exported dataset directory and re-exports the dataset.
"""

import os
import csv
import fiftyone as fo


def main() -> None:
    # Resolve exported dataset directory (relative to project root)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    exported_dataset_dir = os.path.join(project_root, "exported_datasets/exuviae_keypoints")

    # Keep name aligned with existing dataset to preserve prior state
    dataset_name = "exuviae_keypoints"

    print(f"Attempting to load or create dataset '{dataset_name}' from {exported_dataset_dir}...")

    # Load existing dataset if present; otherwise import from directory
    if fo.dataset_exists(dataset_name):
        print(f"Loading existing dataset '{dataset_name}' (preserves tags)...")
        dataset = fo.load_dataset(dataset_name)
    else:
        print(f"Creating dataset '{dataset_name}' from {exported_dataset_dir}...")
        dataset = fo.Dataset.from_dir(
            dataset_dir=exported_dataset_dir,
            dataset_type=fo.types.FiftyOneDataset,
            name=dataset_name,
        )

    print(f"\nLoaded dataset with {len(dataset)} samples")
    print("\nDataset fields:")
    print(dataset.get_field_schema())

    # Launch the app to view/edit the dataset
    print("\nLaunching FiftyOne app...")
    session = fo.launch_app(dataset, port=5173)
    session.wait()

    # After closing the app, export sample-level tags to CSV and re-export dataset
    print("\nExporting sample tags and saving dataset with updated tags...")

    os.makedirs(exported_dataset_dir, exist_ok=True)
    tags_csv_path = os.path.join(exported_dataset_dir, "sample_tags.csv")

    with open(tags_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sample_id", "filepath", "tags"])  # header
        for sample in dataset:
            tag_str = "|".join(sample.tags) if sample.tags else ""
            writer.writerow([str(sample.id), sample.filepath, tag_str])

    print(f"Saved sample tags to {tags_csv_path}")

    dataset.export(
        export_dir=exported_dataset_dir,
        dataset_type=fo.types.FiftyOneDataset,
        export_media=True,
        overwrite=True,
    )

    print(f"Exported updated dataset (with tags) to {exported_dataset_dir}")


if __name__ == "__main__":
    main()


