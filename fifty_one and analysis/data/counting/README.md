## Prawn Counting Dataset

This dataset is designed for training models to detect and count prawns in aquaculture environments.

### Dataset Access

The dataset is publicly available on Roboflow Universe at [Giant Freshwater Prawns Counting](https://universe.roboflow.com/prawns/giant-freshwater-prawns-counting). You can access and download it using:

1. Roboflow Universe web interface at the link above
2. Roboflow Python SDK:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("prawns").project("giant-freshwater-prawns-counting")
dataset = project.version(1).download("yolov8")
```

### Dataset Details

- Collection Date: September 20, 2023
- Source: 9 drone videos + 82 still images
- Location: Ben-Gurion University's Bergman Campus
- Environment: Circular pond housing female prawns
- Conditions: Post-feeding, minimal environmental turbulence
- License: BY-NC-SA 4.0

### Dataset Specifications

- Format: YOLOv8 compatible
- Annotations: Bounding boxes for prawn detection
- Splits: train/valid/test
- Augmentations applied:
  - Rotation: ±15 degrees
  - Brightness adjustment: ±25%
  - Horizontal flip
  - Vertical flip
  - Mosaic augmentation

### Training Parameters

The dataset was trained with the following parameters:
- Model: YOLOv8
- Image size: 640x640
- Batch size: 16
- Learning rate: 0.01
- Epochs: 100

### Environmental Conditions

Images were captured under various conditions:
- Different times of day
- Various water clarity levels
- Multiple pond types
- Different prawn densities

### Best Practices

When using this dataset:
1. Consider the environmental conditions
2. Use appropriate confidence thresholds
3. Validate results against manual counts
4. Account for occlusion and overlap

### Related Scripts

Scripts for processing and analyzing the counting results are available in the repository:
- Count validation
- Density analysis
- Visualization tools

For more details about specific scripts, see the `scripts/` directory.

### External Resources

Pre-trained model weights are available at: https://drive.google.com/drive/folders/1KJ1FOhahT8TRm1uN4n8PedH_XHyj0UDV?usp=sharing 