"""
Extract sample-level tags and detection/keypoint labels from a FiftyOne samples.json and save them to a CSV.

Differences vs run/launch scripts:
- Does not launch FiftyOne or load the dataset; it only parses samples.json.
- Supports both JSON array and newline-delimited JSON (NDJSON) formats.
- Writes a sample_tags.csv with columns: sample_id, filepath, tags, detections.label, labels
  - tags are delimited by |
  - detections.label aggregates labels from Detection/Detections fields
  - labels aggregates Detection/Detections, Keypoint/Keypoints and Classification(s) labels

Output:
- CSV is written next to samples.json as sample_tags.csv by default.

Note: Adjust dataset_dir below if your exported dataset lives elsewhere.
"""

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Set


def _read_samples_json(samples_json_path: str) -> List[Dict[str, Any]]:
    """Reads FiftyOne samples from JSON array or NDJSON file.

    Returns a list of per-sample dicts.
    """
    if not os.path.isfile(samples_json_path):
        raise FileNotFoundError(f"samples.json not found: {samples_json_path}")

    with open(samples_json_path, "r", encoding="utf-8") as f:
        contents = f.read()

    # Detect format: JSON array, JSON object with "samples" key, or NDJSON
    first_non_ws = next((ch for ch in contents.lstrip()[:1]), "")
    if first_non_ws == "[":
        records = json.loads(contents)
        if not isinstance(records, list):
            raise ValueError("Expected a JSON array of samples")
        return records
    if first_non_ws == "{":
        obj = json.loads(contents)
        if isinstance(obj, dict) and "samples" in obj and isinstance(obj["samples"], list):
            return obj["samples"]

    # NDJSON fallback
    records = []
    for line in contents.splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _extract_row(sample: Dict[str, Any]) -> List[str]:
    """Extracts [sample_id, filepath, tags_str, detections_label_str, labels_str] from a sample dict."""
    sample_id = str(sample.get("_id") or sample.get("id") or "")
    filepath = str(sample.get("filepath") or "")
    tags_field = sample.get("tags")

    # Tags may be a list, string, or missing
    if isinstance(tags_field, list):
        tags: List[str] = [str(t) for t in tags_field]
    elif isinstance(tags_field, str):
        tags = [tags_field]
    elif tags_field is None:
        tags = []
    else:
        # Unexpected type; stringify
        tags = [str(tags_field)]

    tag_str = "|".join(tags) if tags else ""

    # Collect labels from Detection(s), Classification(s), and Keypoint(s)
    def collect_labels(value: Any, out_all: Set[str], out_det: Set[str]) -> None:
        if isinstance(value, dict):
            cls = value.get("_cls")
            if cls == "Detections":
                for det in value.get("detections", []) or []:
                    label_val = det.get("label")
                    if isinstance(label_val, str) and label_val:
                        out_all.add(label_val)
                        out_det.add(label_val)
            elif cls == "Detection":
                label_val = value.get("label")
                if isinstance(label_val, str) and label_val:
                    out_all.add(label_val)
                    out_det.add(label_val)
            elif cls == "Classifications":
                for c in value.get("classifications", []) or []:
                    label_val = c.get("label")
                    if isinstance(label_val, str) and label_val:
                        out_all.add(label_val)
            elif cls == "Classification":
                label_val = value.get("label")
                if isinstance(label_val, str) and label_val:
                    out_all.add(label_val)
            elif cls == "Keypoints":
                for kp in value.get("keypoints", []) or []:
                    for name in kp.get("labels", []) or []:
                        if isinstance(name, str) and name:
                            out_all.add(name)
            elif cls == "Keypoint":
                for name in value.get("labels", []) or []:
                    if isinstance(name, str) and name:
                        out_all.add(name)

            # Recurse into nested values
            for nested in value.values():
                collect_labels(nested, out_all, out_det)
        elif isinstance(value, list):
            for item in value:
                collect_labels(item, out_all, out_det)

    labels: Set[str] = set()
    detection_labels: Set[str] = set()
    # Traverse all fields except obvious non-label fields
    for key, val in sample.items():
        if key in {"_id", "id", "filepath", "tags", "metadata", "_media_type", "_rand"}:
            continue
        collect_labels(val, labels, detection_labels)

    detections_label_str = "|".join(sorted(detection_labels)) if detection_labels else ""
    labels_str = "|".join(sorted(labels)) if labels else ""

    return [sample_id, filepath, tag_str, detections_label_str, labels_str]


def main() -> None:
    # Resolve exported dataset directory and key paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    dataset_dir = os.path.join(project_root, "exported_datasets", "exuviae_keypoints")

    samples_json_path = os.path.join(dataset_dir, "samples.json")
    out_csv_path = os.path.join(dataset_dir, "sample_tags.csv")

    print(f"Reading samples from: {samples_json_path}")
    records = _read_samples_json(samples_json_path)
    print(f"Parsed {len(records)} records")

    print(f"Writing tags CSV to: {out_csv_path}")
    with open(out_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sample_id", "filepath", "tags", "detections.label", "labels"])  # header
        for rec in records:
            writer.writerow(_extract_row(rec))

    print("Done.")


if __name__ == "__main__":
    main()


