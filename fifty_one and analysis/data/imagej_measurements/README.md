# Prawn Measurements and Keypoint Detection Datasets

This repository contains two related datasets:

## 1. Keypoint Detection Dataset

The keypoint detection model and dataset is publicly available on Roboflow Universe at [Giant Freshwater Prawn Keypoint Detection](https://universe.roboflow.com/prawns/giant-freshwater-prawn-keypoint-detection-umvh3). This dataset is used for detecting anatomical keypoints on prawns.

### Accessing Keypoint Detection Dataset

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("prawns").project("giant-freshwater-prawn-keypoint-detection-umvh3")
dataset = project.version(1).download("yolov8")
```

## 2. ImageJ Measurements Dataset

This dataset contains the ground truth measurements taken using ImageJ software, focused on length calculation and biomass estimation of giant freshwater prawns (_Macrobrachium rosenbergii_) in aquaculture.

### Accessing the FiftyOne Dataset

The measurements dataset is available in two formats:

1. GitHub Repository (recommended for `body_all` and `carapace_all`):
```python
import fiftyone as fo

# For body measurements
dataset = fo.load_dataset("path/to/body_all")

# For carapace measurements
dataset = fo.load_dataset("path/to/carapace_all")
```

2. Google Drive (for additional datasets):
```bash
# Install gdown if you haven't already
pip install gdown

# Download additional datasets
gdown [YOUR_GOOGLE_DRIVE_FILE_ID]
```

### Dataset Contents

The FiftyOne dataset includes:
- Original images from each pond type (right/female, left/male, car/square)
- Ground truth ImageJ measurements
- Keypoint detection predictions
- Evaluation metrics and IoU scores
- Metadata and measurement information

### Dataset Details

- Source: Images from three pond types with ImageJ measurements
- Location: Ben-Gurion University's Bergman Campus
- Environment: Controlled aquaculture ponds
- Measurements: Manual ImageJ measurements for validation
- Annotations: Corresponding keypoint predictions

### License

These datasets are released under the CC BY 4.0 license. 