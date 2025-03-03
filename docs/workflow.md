# Automated Exuviae Detection and Measurement Workflow

## Overview

This document outlines the complete end-to-end workflow developed for the automated detection, filtering, measurement, and analysis of crayfish exuviae from underwater imagery. The pipeline combines state-of-the-art computer vision techniques with domain-specific algorithms to ensure accurate measurements and reliable data collection for ecological monitoring.

## Component Architecture

The workflow is organized around two primary algorithmic pillars, with shared data processing components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Raw Images    │───▶│  Preprocessing  │───▶│ Enhanced Images │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                                       ▼
┌───────────────────────────────────┐    ┌─────────────────────────────────┐
│        Counting Pipeline          │    │      Measurement Pipeline       │
│  ┌─────────────────────────────┐  │    │  ┌─────────────────────────┐   │
│  │     Object Detection        │◀─┼────┼──┤                         │   │
│  └─────────────┬───────────────┘  │    │  │                         │   │
│                │                   │    │  │                         │   │
│  ┌─────────────▼───────────────┐  │    │  │     Segmentation        │   │
│  │     Object Tracking         │  │    │  │                         │   │
│  └─────────────┬───────────────┘  │    │  │                         │   │
│                │                   │    │  │                         │   │
│  ┌─────────────▼───────────────┐  │    │  └─────────────┬───────────┘   │
│  │      Count Analysis         │  │    │                │                │
│  └─────────────┬───────────────┘  │    │  ┌─────────────▼───────────┐   │
└───────────────┬┴───────────────────┘    │  │    Size Measurement     │   │
                │                          │  └─────────────┬───────────┘   │
                │                          │                │                │
                │                          │  ┌─────────────▼───────────┐   │
                │                          │  │ Statistical Analysis    │   │
                │                          │  └─────────────┬───────────┘   │
                │                          └────────────────┬────────────────┘
                │                                           │
┌───────────────▼───────────────────────────────────────────▼───────────────┐
│                         Integrated Results Analysis                        │
└───────────────────────────────────────────────────────────────────────────┘
```

## Workflow Stages

### 1. Image Collection and Preprocessing

#### Data Collection
- Videos collected from controlled environments (circular and square ponds)
- Frame extraction at optimal intervals to capture distinct molt states
- Quality filtering to remove blurry or obstructed frames

#### Preprocessing
- Lens distortion correction using calibrated camera parameters
- Standardized resizing while preserving aspect ratio
- Lighting normalization to accommodate variable underwater conditions

### 2. Parallel Processing: Counting and Measurement

The core of the workflow consists of two parallel but interconnected pipelines:

#### 2A. Counting Pipeline
The counting component focuses on **detecting and counting instances** of exuviae:

##### Object Detection
- YOLOv8 architecture selected for real-time performance and accuracy
- Custom dataset creation with 2,500+ annotated images
- Transfer learning from pre-trained weights
- Data augmentation to improve robustness to underwater conditions

##### Object Tracking (Optional)
- Frame-to-frame tracking for video sequences
- Object persistence verification
- Avoiding duplicate counting

##### Count Analysis
- Population density calculation
- Spatial distribution mapping
- Temporal analysis of count changes

#### 2B. Measurement Pipeline
The measurement component focuses on **determining the size** of detected exuviae:

##### Image Segmentation
- Implementation of specialized color-based segmentation to enhance exuviae visibility
- Background/foreground separation using adaptive thresholding
- Color transformation to highlight key features:
  * Background areas transformed to turquoise (BGR: 31, 156, 212)
  * Potential exuviae areas rendered in gray-brown (BGR: 79, 66, 52)

##### Size Measurement
- Bounding box extraction for each detected exuviae
- Major and minor axis measurement
- Area calculation for comprehensive size estimation

##### Statistical Analysis
- Size distribution visualization
- Temporal growth tracking
- Population health indicator extraction

### 3. Integration Points

The counting and measurement pipelines interact at several key points:

1. **Shared Detection**: The object detection results from the counting pipeline serve as input to the measurement pipeline
2. **Validation Feedback**: Measurement results can validate detection quality
3. **Combined Analysis**: Population counts and size distributions combine for comprehensive ecological insights

### 4. Validation and Quality Control

#### Manual Verification
- Random sampling of detection results for expert validation
- Error analysis and model refinement
- Precision-recall evaluation across different environmental conditions

#### Performance Metrics
- Mean Average Precision (mAP): 0.89
- F1 Score: 0.92
- Average measurement error: ±1.2mm

## Technical Implementation

The entire workflow is implemented as a modular Python framework with the following components:

```
src/
├── counting/       # Detection and counting algorithms
│   ├── detection/  # Object detection models
│   └── tracking/   # Object tracking algorithms
├── measurement/    # Size measurement utilities
│   ├── segmentation/  # Image segmentation algorithms
│   ├── keypoints/     # Keypoint detection
│   └── calibration/   # Camera calibration utilities
├── data/           # Data processing and augmentation
└── utils/          # General utility functions
```

## Future Improvements

- Integration of tracking for individual identification
- Temporal analysis across multiple molt cycles
- Deployment optimization for edge computing devices

## References

1. [Reference papers and technical documentation] 