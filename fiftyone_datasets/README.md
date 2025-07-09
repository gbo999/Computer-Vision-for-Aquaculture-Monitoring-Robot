# FiftyOne Datasets for Prawn Measurements

This repository contains two FiftyOne datasets for visualizing and analyzing prawn measurements:

1. `body_all/` - Full body measurements using combined weights
2. `carapace_all/` - Carapace measurements using combined weights

## Using the Datasets

1. Install FiftyOne:
```bash
pip install fiftyone
```

2. Load a dataset:
```python
import fiftyone as fo

# For body measurements
dataset = fo.load_dataset("path/to/body_all")

# For carapace measurements
dataset = fo.load_dataset("path/to/carapace_all")

# Launch the FiftyOne UI for visualization
session = fo.launch_app(dataset)
session.wait()
```

## Dataset Contents

Each dataset includes:
- Original images from each pond type (right/female, left/male, car/square)
- Ground truth ImageJ measurements
- Keypoint detection predictions
- Evaluation metrics and IoU scores
- Metadata and measurement information

## Related Resources

- Roboflow Universe Projects:
  - [Giant Freshwater Prawn Keypoint Detection](https://universe.roboflow.com/prawns/giant-freshwater-prawn-keypoint-detection-umvh3)
  - [Giant Freshwater Prawns Counting](https://universe.roboflow.com/prawns/giant-freshwater-prawns-counting)

## License

These datasets are released under the CC BY 4.0 license. 