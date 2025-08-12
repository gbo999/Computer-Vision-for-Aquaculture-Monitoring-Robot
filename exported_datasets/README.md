# Exported Datasets Directory

This directory contains **complete FiftyOne datasets** that have been exported from the FiftyOne platform. These are full, self-contained datasets that can be loaded directly into FiftyOne for analysis, visualization, and machine learning workflows.

## Overview

The `exported_datasets` directory contains 4 distinct datasets, each focused on different aspects of prawn research:

1. **Body Measurement Dataset** (`body_all/`) - 71 images with pose estimation and body measurements
2. **Carapace Measurement Dataset** (`carapace_all/`) - 71 images with carapace-specific measurements  
3. **Exuviae Keypoints Dataset** (`exuviae_keypoints/`) - 144 images with keypoint annotations for exuviae analysis
4. **Prawn Counting Dataset** (`prawn_counting/`) - 675 images for prawn detection and counting tasks

## Dataset Details

### 1. Body Measurement Dataset (`body_all/`)

**Purpose**: Comprehensive body measurement analysis of prawns using pose estimation and geometric measurements.

**Key Features**:
- **71 images** of prawns with full body annotations
- **Pose estimation** with 4 keypoints: start_carapace, eyes, rostrum, tail
- **Multiple measurement lines**: 6 different diagonal measurements (max, mid, min for both directions)
- **Environmental metadata**: pond information, lighting conditions, height measurements
- **Evaluation results**: COCO-style pose evaluation with IoU metrics

**Annotations Include**:
- `keypoints`: Pose estimation with confidence scores
- `keypoints_truth`: Ground truth pose annotations
- `ground_truth`: Detection bounding boxes
- `detections_predictions`: Model predictions
- `max_diagonal_line_1/2`, `mid_diagonal_line_1/2`, `min_diagonal_line_1/2`: Measurement polylines
- Environmental data: `pond`, `avg speed (km/h)`, `luminance(cd/m²)`, `height(mm)`, `second in video`

**Use Cases**:
- Body size analysis and growth tracking
- Pose estimation model training and evaluation
- Environmental factor correlation studies
- Measurement accuracy validation

### 2. Carapace Measurement Dataset (`carapace_all/`)

**Purpose**: Specialized carapace measurement analysis with the same structure as body_all but focused on carapace-specific measurements.

**Key Features**:
- **71 images** with carapace-focused annotations
- **Identical structure** to body_all dataset
- **Carapace-specific measurements** and analysis
- **Same pose estimation framework** with 4 keypoints
- **Environmental metadata** included

**Use Cases**:
- Carapace size and shape analysis
- Carapace growth pattern studies
- Comparison with body measurements
- Carapace-specific model development

### 3. Exuviae Keypoints Dataset (`exuviae_keypoints/`)

**Purpose**: Analysis of prawn exuviae (molted shells) using keypoint detection and bounding box annotations.

**Key Features**:
- **144 images** of prawn exuviae
- **Keypoint annotations** with 4-point skeleton: start_carapace, eyes, rostrum, tail
- **Bounding box detections** for exuviae objects
- **Custom polylines**: `shai_polyline1` and `shai_polyline2` for specialized measurements
- **Detection annotations** for object localization

**Annotations Include**:
- `keypoints`: Exuviae keypoint annotations
- `detections`: General detection bounding boxes
- `bounding_box`: Specific bounding box annotations
- `shai_polyline1/2`: Custom measurement polylines
- `name`: Sample identification

**Use Cases**:
- Exuviae size and shape analysis
- Molting pattern studies
- Exuviae detection model training
- Growth stage classification

### 4. Prawn Counting Dataset (`prawn_counting/`)

**Purpose**: Large-scale prawn detection and counting dataset for population analysis.

**Key Features**:
- **675 images** - the largest dataset in this collection
- **Detection annotations**: Ground truth and model predictions
- **Evaluation metrics**: True positives, false positives, false negatives
- **COCO-style evaluation** with IoU thresholds
- **Population counting** focus

**Annotations Include**:
- `ground_truth`: Manual prawn detections
- `prawn`: Model prediction detections
- `eval_tp/fp/fn`: Evaluation metrics per image
- `eval`, `eval_id`, `eval_iou`: Evaluation results

**Use Cases**:
- Population density estimation
- Prawn counting automation
- Detection model training and evaluation
- Pond population monitoring

## Dataset Structure

Each dataset follows the standard FiftyOne export structure:

```
dataset_name/
├── data/                    # Image files (.jpg)
├── evaluations/            # Evaluation results (if applicable)
├── fields/                 # Additional field data (if applicable)
├── metadata.json          # Dataset metadata and schema
└── samples.json           # Sample annotations and data
```

## Loading Datasets

To load any of these datasets into FiftyOne:

```python
import fiftyone as fo

# Load a dataset
dataset = fo.Dataset.from_dir(
    "exported_datasets/body_all",
    dataset_type=fo.types.FiftyOneDataset
)

# View dataset info
print(dataset)
print(f"Number of samples: {len(dataset)}")

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

## Key Differences Between Datasets

| Dataset | Images | Focus | Key Annotations | Primary Use |
|---------|--------|-------|-----------------|-------------|
| `body_all` | 71 | Body measurements | Pose + 6 measurement lines | Size analysis |
| `carapace_all` | 71 | Carapace measurements | Pose + 6 measurement lines | Carapace analysis |
| `exuviae_keypoints` | 144 | Exuviae analysis | Keypoints + polylines | Molting studies |
| `prawn_counting` | 675 | Population counting | Detections + eval metrics | Population monitoring |

## Environmental Metadata

The body and carapace datasets include rich environmental metadata:
- **Pond identification**: Source pond information
- **Lighting conditions**: Luminance measurements in cd/m²
- **Height measurements**: Camera height in mm
- **Temporal data**: Video timestamps and frame information
- **Speed data**: Average movement speed in km/h

## Evaluation Results

Several datasets include evaluation results:
- **Pose evaluation**: COCO-style keypoint evaluation with IoU metrics
- **Detection evaluation**: True positive, false positive, false negative counts
- **IoU thresholds**: Multiple intersection-over-union thresholds for comprehensive evaluation

## Usage Recommendations

1. **For measurement studies**: Use `body_all` or `carapace_all`
2. **For population analysis**: Use `prawn_counting`
3. **For molting research**: Use `exuviae_keypoints`
4. **For model training**: All datasets can be used depending on the task

## Data Quality Notes

- All images are undistorted and gamma-corrected
- Annotations include confidence scores where applicable
- Environmental conditions are documented for reproducibility
- Evaluation metrics provide quantitative performance measures

## Related Files

These datasets are used in conjunction with:
- Analysis scripts in `fifty_one_and_analysis/`
- Training outputs in `training and val output/`
- Archived spreadsheets in `archived_spreadsheets/`
- Visualization tools throughout the project

---

**Note**: These are complete, self-contained FiftyOne datasets. Each can be loaded independently and contains all necessary metadata, annotations, and evaluation results for comprehensive analysis. 