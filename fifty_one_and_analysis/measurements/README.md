# Prawn Measurement Analysis System

## Overview

This directory contains a comprehensive measurement analysis system for prawn research using computer vision, ImageJ measurements, and FiftyOne visualization. The system processes both live prawn measurements (body and carapace) and exuviae (molted shells) measurements, providing automated detection, validation, and statistical analysis capabilities.

## 🎯 Research Objectives

The measurement analysis system addresses several key research challenges in prawn aquaculture and marine biology:

1. **Automated Length Measurement**: Replace manual measurement with computer vision-based detection
2. **Measurement Validation**: Compare automated predictions against ground truth measurements
3. **Error Analysis**: Quantify and analyze measurement accuracy across different conditions
4. **Data Visualization**: Interactive exploration of measurement data and predictions
5. **Exuviae Analysis**: Specialized processing for molted shell measurements

## 📁 Directory Structure

```
measurements/
├── README.md                           # This documentation file
├── imagej/                             # Live prawn measurement analysis
│   ├── README.md                       # Detailed ImageJ analysis documentation
│   ├── run_fiftyone_body.py           # FiftyOne launcher for body measurements
│   ├── run_fiftyone_carapace.py       # FiftyOne launcher for carapace measurements
│   ├── Imagej_analysis/               # Core analysis scripts and notebooks
│   ├── spreadsheet_files/             # Measurement data and results
│   └── images/                        # Visualization examples
├── data_processing_scripts/           # Data preprocessing and video processing
│   ├── README.md                      # Data processing documentation
│   ├── undistort_images.py           # GoPro image undistortion
│   ├── imagej- data exctraction/     # ImageJ measurement processing
│   └── video_processing/             # Video-to-dataset conversion
└── exuviae/                           # Exuviae (molted shell) analysis
    ├── README.md                      # Exuviae analysis documentation
    ├── 0-binary_exuviae_colorizer.py # Image preprocessing and colorization
    ├── 1-size_classifier.py          # Size classification
    ├── 2-split_rows_by_size.py       # Dataset splitting by size
    ├── 3-merging_imagej_images.ipynb # Manual vs automated comparison
    ├── 4-bbox_compare_prediction_with_manual_pixels_exuviae.py # Bounding box validation
    ├── 5-Exuviae_analysis.ipynb      # Main analysis notebook
    ├── create_fiftyone_exuviae_keypoints_dataset.py # FiftyOne dataset creation
    ├── run_exuviae_fiftyone.py       # FiftyOne execution script
    ├── archived/                     # Legacy scripts and data
    ├── images/                       # Analysis visualizations
    └── spreadsheet_files/            # Analysis results and data
```

## 🔄 Analysis Workflows

### 1. Live Prawn Measurement Analysis (imagej/)

**Purpose**: Analyze body and carapace measurements of live prawns using ImageJ data and YOLO keypoint detection.

**Workflow**:
1. **Data Collection**: Manual measurements using ImageJ
2. **Automated Detection**: YOLO keypoint detection on images
3. **Validation**: Compare automated predictions with ground truth
4. **Error Analysis**: Calculate MAE, MAPE, and position errors
5. **Visualization**: Interactive FiftyOne interface for data exploration

**Key Features**:
- Support for multiple measurement types (carapace, body)
- Multiple prediction weights (car, kalkar, all)
- Interactive error analysis with filtering
- Real-time visualization with FiftyOne

**Usage**:
```bash
# Launch body measurement analysis
cd imagej/
python run_fiftyone_body.py

# Launch carapace measurement analysis  
python run_fiftyone_carapace.py
```

### 2. Data Processing Pipeline (data_processing_scripts/)

**Purpose**: Preprocess and prepare data for measurement analysis, including image undistortion and video processing.

**Components**:
- **Image Undistortion**: Remove GoPro lens distortion using calibrated parameters
- **ImageJ Data Extraction**: Process manual measurement data and assign consistent IDs
- **Video Processing**: Convert video files to image datasets with quality filtering

**Key Features**:
- GoPro Hero 11 specific camera calibration
- Spatial proximity matching for prawn ID assignment
- Statistical analysis of measurement consistency
- Quality-based video frame extraction

**Usage**:
```bash
# Undistort GoPro images
cd data_processing_scripts/
python undistort_images.py

# Process ImageJ measurements
cd imagej- data exctraction/
python 1-prawn_measurement_id_assignment.py
python 2- from imagej-prawn_measurement_statistics.py
```

### 3. Exuviae Analysis (exuviae/)

**Purpose**: Specialized analysis of prawn exuviae (molted shells) for size classification and measurement validation.

**Workflow**:
1. **Image Preprocessing**: Colorize white molts for better detection
2. **Size Classification**: Categorize exuviae by size (big: 165-220mm, small: 116-164mm)
3. **Dataset Splitting**: Separate datasets by size category
4. **Validation**: Compare YOLO predictions with manual measurements
5. **Statistical Analysis**: Calculate error metrics and generate reports

**Key Features**:
- Binary thresholding for molt identification
- Realistic colorization for improved detection
- Size-based dataset organization
- Comprehensive error analysis

**Usage**:
```bash
# Process exuviae images
cd exuviae/
python 0-binary_exuviae_colorizer.py
python 1-size_classifier.py
python 2-split_rows_by_size.py

# Launch FiftyOne visualization
python run_exuviae_fiftyone.py
```

## 🛠️ Technical Implementation

### What I Did Differently

**Modular Architecture**: I implemented a modular system where each component has a specific responsibility and can be used independently or as part of the larger pipeline.

**Robust Error Handling**: Each script includes comprehensive error handling and validation to ensure reliable processing of large datasets.

**Dynamic Configuration**: The system uses command-line arguments and configuration files to support different measurement types and prediction weights without code changes.

**Interactive Visualization**: Integration with FiftyOne provides interactive data exploration capabilities that go beyond static plots and tables.

### Key Technical Features

1. **Camera Calibration**: GoPro Hero 11 specific undistortion using calibrated parameters
2. **Spatial Matching**: Algorithm for matching prawns across multiple measurement sessions
3. **Quality Filtering**: Automated quality assessment for video frame extraction
4. **Error Metrics**: Comprehensive error analysis including MAE, MAPE, and position errors
5. **Dataset Management**: Automated FiftyOne dataset creation and management

## 📊 Data Formats

### Input Data
- **Images**: JPG files from GoPro cameras (with optional undistortion)
- **Measurements**: Excel/CSV files from ImageJ manual measurements
- **Predictions**: YOLO keypoint detection results
- **Video**: MP4 files for video-to-dataset conversion

### Output Data
- **Analysis Results**: CSV files with error metrics and statistical summaries
- **Visualizations**: Interactive FiftyOne datasets and static plots
- **Processed Images**: Undistorted and colorized images for analysis
- **Reports**: Comprehensive analysis notebooks and documentation

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- OpenCV
- FiftyOne
- NumPy, Pandas
- Jupyter Notebook (for analysis notebooks)

### Installation
```bash
# Install required packages
pip install fiftyone opencv-python numpy pandas jupyter

# Clone the repository and navigate to measurements folder
cd "fifty_one_and_analysis/measurements"
```

### Quick Start
1. **For Live Prawn Analysis**:
   ```bash
   cd imagej/
   python run_fiftyone_body.py
   ```
   or 
   ```bash
   cd imagej/
   python run_fiftyone_carapace.py
   ```

2. **For Exuviae Analysis**:
   ```bash
   cd exuviae/
   python run_exuviae_fiftyone.py
   ```


## 🔍 Troubleshooting

### Common Issues
1. **Port Conflicts**: FiftyOne launchers use random port assignment to avoid conflicts
2. **Memory Issues**: Large datasets may require batch processing
3. **Camera Calibration**: Ensure GoPro calibration parameters are up to date

### Support
- Check individual README files in each subdirectory for detailed documentation
- Review error logs and console output for specific issues
- Use FiftyOne's built-in debugging tools for visualization issues

## 📚 References

- **FiftyOne Documentation**: https://docs.voxel51.com/
- **OpenCV Camera Calibration**: https://docs.opencv.org/
- **YOLO Keypoint Detection**: https://github.com/ultralytics/ultralytics
- **ImageJ Measurement**: https://imagej.nih.gov/ij/

## 👥 Contributing

When adding new scripts or modifying existing ones:
1. Follow the established naming conventions
2. Include comprehensive documentation
3. Add error handling and validation
4. Update this README with new components
5. Test with sample data before deployment

---

**Author**: Gil Benor  
**Last Updated**: December 2024  
**Version**: 1.0 