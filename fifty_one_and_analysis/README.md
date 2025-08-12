# FiftyOne Analysis Framework for Prawn Research

This directory contains comprehensive analysis tools using FiftyOne for prawn research, including measurement analysis and counting detection. The framework provides interactive visualization and analysis capabilities for computer vision research in aquaculture.

## 📁 Directory Structure

```
fifty_one_and_analysis/
├── README.md                           # This documentation file
├── DATA_ACCESS.md                      # Dataset access and storage information
├── counting/                           # Prawn counting analysis
│   ├── README.md                       # Counting analysis documentation
│   ├── run_fiftyone_counting.py       # FiftyOne launcher for counting dataset
│   └── counting-fiftyone_dataset_creation_and_analysis.ipynb # Main analysis notebook
└── measurements/                       # Prawn measurement analysis
    ├── README.md                       # Measurement analysis documentation
    ├── imagej/                         # Live prawn measurement analysis
    ├── data_processing_scripts/        # Data preprocessing and video processing
    └── exuviae/                        # Exuviae (molted shell) analysis
```

## 🎯 Research Areas

### 1. **Prawn Counting Analysis** (`counting/`)
- **Purpose**: Analyze prawn counting accuracy in underwater imagery
- **Features**: 
  - YOLO-based detection model evaluation
  - Multi-threshold confidence analysis
  - Interactive FiftyOne visualization
  - Performance metrics and error analysis
- **Dataset**: Uses `exported_datasets/prawn_counting/` for model evaluation

### 2. **Prawn Measurement Analysis** (`measurements/`)
- **Purpose**: Automated length measurement and validation
- **Components**:
  - **Live Prawn Measurements** (`imagej/`): Body and carapace measurements using ImageJ data
  - **Data Processing** (`data_processing_scripts/`): Image undistortion and video processing
  - **Exuviae Analysis** (`exuviae/`): Molted shell measurement analysis

## 🚀 Quick Start

### Prerequisites

1. **Install FiftyOne**:
   ```bash
   pip install fiftyone
   ```

2. **Install additional dependencies**:
   ```bash
   pip install ultralytics pillow pytesseract matplotlib seaborn opencv-python
   ```


### Running Analysis

#### Counting Analysis
```bash
cd counting/
python run_fiftyone_counting.py
```

#### Measurement Analysis
```bash
cd measurements/imagej/
python run_fiftyone_body.py      # For body measurements
python run_fiftyone_carapace.py  # For carapace measurements
```

#### Exuviae Analysis
```bash
cd measurements/exuviae/
python run_exuviae_fiftyone.py
```

## 📊 Dataset Requirements

### Counting Dataset
- **Location**: `exported_datasets/prawn_counting/`
- **Contents**: Underwater images, bounding box annotations, ground truth labels


### Measurement Datasets
- **ImageJ Measurements**: Live prawn measurement images 
- **Molt/Exuviae**: Processed molt images 
- **Drone Detection**: Aerial imagery and detection results 

## 🔧 Configuration

### FiftyOne Settings
- **Dataset Management**: Automatic dataset loading and cleanup
- **Visualization**: Interactive web interface for data exploration
- **Performance**: Optimized for large-scale image analysis

### Model Integration
- **YOLO Models**: RT-DETR and YOLOv8 for detection and keypoint estimation
- **Evaluation**: Multi-threshold confidence analysis
- **Validation**: Ground truth comparison and error metrics

## 📈 Analysis Capabilities

### Interactive Visualization
- **Image Browsing**: Navigate through large image datasets
- **Detection Overlay**: Visualize model predictions and ground truth
- **Filtering**: Filter by confidence, error type, or measurement criteria
- **Statistics**: Real-time performance metrics and error analysis

### Automated Processing
- **Batch Processing**: Handle large datasets efficiently
- **Error Analysis**: Calculate MAE, MAPE, and position errors
- **Data Export**: Generate reports and visualizations
- **Quality Control**: Identify and flag problematic samples

## 📚 Documentation

- **`DATA_ACCESS.md`**: Complete guide for accessing and managing datasets
- **`counting/README.md`**: Detailed counting analysis documentation
- **`measurements/README.md`**: Comprehensive measurement analysis guide

## 🔍 Key Features

1. **Multi-Modal Analysis**: Support for counting, measurement, and exuviae analysis
2. **Interactive Interface**: FiftyOne web interface for data exploration
3. **Automated Validation**: Compare automated predictions with ground truth
4. **Error Analysis**: Comprehensive error metrics and visualization
5. **Scalable Architecture**: Handle large-scale datasets efficiently

