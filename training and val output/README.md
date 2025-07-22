# Training and Validation Output Directory

This directory contains all outputs from YOLO model training, validation, and prediction runs for the counting research algorithms project. The folder structure organizes different types of model outputs and experimental results.

## Directory Structure

### 📁 **runs/**
Contains the main training and prediction outputs organized by model type and run number.

#### **runs/detect/**
Detection model outputs with validation results:
- `val/` through `val8/` - Validation runs with prediction images and metrics
- Each validation folder contains:
  - `val_batch*_pred.jpg` - Model predictions on validation batches
  - `val_batch*_labels.jpg` - Ground truth labels for comparison
  - `predictions.json` - Detailed prediction results and metrics

#### **runs/pose/**
Pose estimation model outputs with extensive prediction runs:
- `val/` through `val21/` - Validation runs
- `predict2/` through `predict90/` - Prediction runs on test data
- Each folder contains annotated images with keypoint detections

#### **runs/predict/**
General prediction outputs from various model runs

### 📁 **labels/**
Contains YOLO format annotation files for training data:
- Files named as `undistorted_GX010191_10_370.txt` format
- Each file contains bounding box and keypoint annotations in YOLO format:
  - Class ID (0 for prawn)
  - Bounding box coordinates (x_center, y_center, width, height) - normalized
  - Keypoint coordinates (x, y, confidence) for multiple keypoints
  - Format: `class_id x_center y_center width height kp1_x kp1_y kp1_conf kp2_x kp2_y kp2_conf ...`

### 📁 **test-car/**, **test-left/**, **test-right/**
Test datasets organized by camera position/orientation:
- `images/` - Test images with various naming conventions
- `labels/` - Corresponding annotation files
- Used for evaluating model performance on different viewing angles

### 📁 **vals/** and **vals2/**
Additional validation outputs:
- `val/` through `val10/` - Multiple validation runs
- Contains prediction vs ground truth comparison images
- `predictions.json` files with detailed metrics

## File Types and Formats

### Image Files
- **Prediction Images**: `*_pred.jpg` - Model predictions with bounding boxes and keypoints
- **Label Images**: `*_labels.jpg` - Ground truth annotations for comparison
- **Test Images**: Various formats including undistorted and gamma-corrected versions

### Annotation Files
- **YOLO Format**: `.txt` files with normalized coordinates
- **JSON Files**: `predictions.json` with detailed prediction results and metrics

### Key Features
- **Multi-class detection**: Prawn detection with bounding boxes
- **Keypoint detection**: Multiple anatomical keypoints with confidence scores
- **Pose estimation**: Full pose estimation capabilities
- **Validation metrics**: Comprehensive evaluation results

## Usage Notes

1. **Model Comparison**: Use validation folders to compare different model versions
2. **Performance Analysis**: Check `predictions.json` files for detailed metrics
3. **Visual Inspection**: Compare `*_pred.jpg` vs `*_labels.jpg` for qualitative assessment
4. **Test Evaluation**: Use test folders for final model evaluation on unseen data

## Naming Conventions

- **Images**: `undistorted_GX010191_10_370.jpg` - Undistorted images with frame numbers
- **Labels**: Matching `.txt` files with same base name
- **Runs**: Sequential numbering (val1, val2, predict1, predict2, etc.)
- **Batches**: `val_batch0_pred.jpg` - Validation batch predictions

## Model Types

1. **Detection Model**: Object detection with bounding boxes
2. **Pose Model**: Keypoint detection and pose estimation
3. **Combined Model**: Both detection and pose estimation capabilities

This directory serves as the central repository for all model training outputs, enabling comprehensive analysis of model performance across different datasets and experimental conditions. 