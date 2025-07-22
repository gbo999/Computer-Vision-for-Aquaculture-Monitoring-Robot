# ImageJ Measurement Analysis with FiftyOne

## Overview

This directory contains a comprehensive measurement analysis system that combines ImageJ measurement data with FiftyOne visualization and YOLO keypoint detection for prawn length analysis. The system processes both carapace and body measurements, validates predictions against ground truth, and provides detailed error analysis with interactive visualizations.

## FiftyOne Visualization Examples

### Carapace Length vs Ground Truth
![Carapace Length vs Ground Truth](images/carapace%20lengtj%20vs%20ground%20truth.png)

*Example of FiftyOne visualization showing carapace length measurements compared to ground truth values. This demonstrates the interactive measurement validation interface.*

### Total Length Measurement Example
![Total Length Example](images/total%20length%20example.png)

*Example of FiftyOne visualization showing total length measurements with keypoint detection and measurement overlays. This illustrates the comprehensive measurement analysis capabilities.*

## Directory Structure

```
imagej/
├── README.md                                    # This file
├── run_fiftyone_body.py                        # FiftyOne launcher for body measurements
├── run_fiftyone_carapace.py                    # FiftyOne launcher for carapace measurements
├── Imagej_analysis/                            # Core analysis scripts
│   ├── fiftyone_dataset_creation.py            # Main dataset creation script
│   ├── fiftyone_dataset_backend.py             # Backend processing engine
│   ├── error_flags_analysis_and_filtered_dataset.py  # filtering and different metrics
│   ├── measurement_analysis_notebook.ipynb     # Jupyter notebook for analysis of results
│   ├── utils.py                                # Utility functions
│   ├── run_analysis.sh                         # Shell script for analysis pipeline
│   ├── fifty_one.sh                            # FiftyOne dataset setup shell script
│   └── fiftyone_measurements.spec              # PyInstaller specification
└── spreadsheet_files/                          # Data files
    ├── Filtered_Data.csv                       # Filtered measurement data
    ├── error_flags_analysis_*.csv              # Error analysis results
    ├── updated_filtered_data_with_lengths_*.xlsx  # Processed measurement data
    └── test images.xlsx                        # Image metadata
```

## Scripts Documentation

### 1. FiftyOne Dataset Launchers

#### `run_fiftyone_body.py`
**Purpose**: Launches FiftyOne visualization for body measurement datasets.

**What I did differently**: I implemented a robust dataset loading system that automatically handles existing datasets by deleting and recreating them to ensure clean visualization. The script uses random port assignment to avoid conflicts: `port=random.randint(10000, 65535)`.

**Key Features**:
- Automatic dataset cleanup and recreation
- Random port assignment for conflict avoidance
- Dataset schema display
- Interactive FiftyOne app launch

**Usage**:
```bash
python run_fiftyone_body.py
```

#### `run_fiftyone_carapace.py`
**Purpose**: Launches FiftyOne visualization for carapace measurement datasets.

**What I did differently**: Similar to body launcher but specifically configured for carapace datasets with different path handling and dataset naming conventions.

**Key Features**:
- Carapace-specific dataset loading
- Automatic dataset management
- Interactive visualization interface

**Usage**:
```bash
python run_fiftyone_carapace.py
```

---

### 2. Core Analysis Scripts

#### `Imagej_analysis/fiftyone_dataset_creation.py`
**Purpose**: Main orchestration script for creating FiftyOne datasets from ImageJ measurements and YOLO predictions.

**What I did differently**: I implemented a modular dataset creation system that supports multiple measurement types (carapace/body) and prediction versions (car/kalkar/all). The script uses dynamic path resolution and port availability checking to ensure robust execution.

**Key Features**:
- Support for multiple measurement types (carapace, body)
- Multiple prediction weights support (car, kalkar, all)
- Dynamic path resolution based on weights type
- Port availability checking and assignment
- Modular data loading and dataset creation

**Command Line Arguments**:
```bash
python fiftyone_dataset_creation.py --type carapace --weights_type all --port 5159
```

**Processing Pipeline**:
1. Parse command line arguments
2. Resolve paths based on weights type
3. Load filtered data and metadata
4. Create FiftyOne dataset
5. Process images and predictions
6. Launch interactive visualization

#### `Imagej_analysis/fiftyone_dataset_backend.py`
**Purpose**: Backend processing engine for creating and managing FiftyOne datasets with real-world measurement calculations.

**What I did differently**: I implemented a sophisticated real-world measurement system using camera calibration parameters and FOV calculations. The `ObjectLengthMeasurer` class uses trigonometric scaling factors to convert pixel measurements to real-world distances: `scale_x = (2 * distance_mm * math.tan(fov_x_rad / 2)) / image_width`.

**Key Classes**:

##### `ObjectLengthMeasurer`
**What I did differently**: Implements camera-aware measurement conversion using:
- **FOV-based scaling**: Calculates mm/pixel ratios using camera field of view
- **Angle normalization**: Normalizes angles to [0°, 90°] range for consistent calculations
- **Combined scaling**: Uses both horizontal and vertical scaling factors based on measurement angle

```python
def calculate_scaling_factors(self):
    fov_x_rad = math.radians(self.horizontal_fov)
    fov_y_rad = math.radians(self.vertical_fov)
    scale_x = (2 * self.distance_mm * math.tan(fov_x_rad / 2)) / self.image_width
    scale_y = (2 * self.distance_mm * math.tan(fov_y_rad / 2)) / self.image_height
    return scale_x, scale_y
```

**Key Functions**:
- `load_data()`: Loads carapace measurement data
- `load_data_body()`: Loads body measurement data
- `create_dataset()`: Creates FiftyOne dataset for carapace measurements
- `create_dataset_body()`: Creates FiftyOne dataset for body measurements
- `process_poses()`: Processes YOLO keypoint predictions
- `add_metadata()`: Adds measurement metadata to samples
- `process_images()`: Processes images and predictions

**Camera Specifications**:
- **Resolution**: 5312x2988 pixels
- **Camera**: GoPro Hero 11
- **FOV**: Configurable horizontal and vertical field of view
- **Distance**: Configurable measurement distance

#### `Imagej_analysis/error_flags_analysis_and_filtered_dataset.py`
**Purpose**: Comprehensive error analysis and visualization system for measurement validation.

**What I did differently**: I implemented a multi-flag error detection system that analyzes measurement errors from multiple perspectives. The system uses MAPE (Mean Absolute Percentage Error) calculations and creates interactive visualizations using both Matplotlib/Seaborn and Plotly for comprehensive analysis.

**Key Features**:
- **Multi-flag error detection**: Analyzes errors from multiple angles
- **Statistical analysis**: Calculates MAPE, R² scores, and correlation coefficients
- **Interactive visualizations**: Uses Plotly for dynamic charts
- **Pond-type analysis**: Separates analysis by pond type (square, circle_female, circle_male)
- **Error categorization**: Categorizes errors by type and severity

**Error Metrics**:
```python
def calculate_mape(estimated_lengths, true_lengths):
    absolute_percentage_errors = [abs(est - true) / est * 100 
                                 for est, true in zip(estimated_lengths, true_lengths)]
    return absolute_percentage_errors
```

**Visualization Types**:
- Scatter plots of predicted vs true measurements
- Error distribution histograms
- Correlation heatmaps
- Box plots by pond type
- Interactive 3D scatter plots

**Usage**:
```bash
python error_flags_analysis_and_filtered_dataset.py --type carapace --weights_type all --error_size mean
```

#### `Imagej_analysis/utils.py`
**Purpose**: Utility functions for YOLO pose estimation parsing and geometric calculations.

**What I did differently**: I implemented a robust YOLO pose estimation parser that handles the complex YOLOv8 keypoint format with comprehensive error checking and validation. The parser validates keypoint triplets, value ranges, and data integrity.

**Key Functions**:

##### `parse_pose_estimation()`
**What I did differently**: Implements comprehensive validation for YOLO keypoint format:
- **Keypoint validation**: Ensures complete keypoint triplets (x, y, confidence)
- **Value range checking**: Validates all values are in [0,1] range
- **Duplicate detection**: Prevents duplicate lines in output
- **Error handling**: Graceful handling of malformed data

```python
def parse_pose_estimation(txt_file: str) -> List[List[float]]:
    # Validates YOLOv8 format: class xc yc w h kp1x kp1y kp1v kp2x kp2y kp2v ...
    # Keypoint order: start_carapace, eyes, rostrum, tail
```

**Keypoint Format**:
- **Class ID**: 0 for prawn
- **Bounding Box**: xc, yc, w, h (normalized [0-1])
- **Keypoints**: kpNx, kpNy, kpNv (coordinates + confidence)
- **Keypoint Order**: start_carapace, eyes, rostrum, tail

**Additional Functions**:
- `calculate_euclidean_distance()`: Calculates distance between two points
- `extract_identifier_from_gt()`: Extracts identifiers from ground truth files
- `calculate_bbox_area()`: Calculates bounding box area

---

### 3. Shell Scripts

#### `Imagej_analysis/run_analysis.sh`
**Purpose**: Shell script for running the complete analysis pipeline.

**What I did differently**: I created a comprehensive shell script that orchestrates the entire analysis workflow, including dataset creation, error analysis, and visualization generation.

#### `Imagej_analysis/fifty_one.sh`
**Purpose**: FiftyOne setup and configuration script.

**What I did differently**: Implements FiftyOne-specific setup and configuration for the measurement analysis environment.

---

### 4. Data Files

#### `spreadsheet_files/`
**Purpose**: Contains all measurement data and analysis results.

**Key Files**:
- **Filtered_Data.csv**: Base filtered measurement data
- **error_flags_analysis_*.csv**: Error analysis results by measurement type
- **updated_filtered_data_with_lengths_*.xlsx**: Processed measurement data with calculated lengths
- **test images.xlsx**: Image metadata and calibration information

**Data Structure**:
- **Measurement Types**: carapace, body
- **Pond Types**: car (square), right (circle_female), left (circle_male)
- **Prediction Versions**: all, kalkar, car

---

## Quick Start - Commands to Run

### 1. Install Dependencies
```bash
pip install fiftyone pandas numpy matplotlib seaborn plotly scikit-learn scipy
fiftyone brain init
```

### 2. Create Directories
```bash
mkdir -p exported_datasets/body_all exported_datasets/carapace_all
mkdir -p fiftyone_datasets
```

### 3. Run Analysis Pipeline
```bash
# Navigate to the analysis directory
cd "fifty_one and analysis/measurements/imagej/Imagej_analysis"

# Option 1: Use the shell script for dataset creation
./fifty_one.sh

# Option 2: Run individual commands
# Create carapace dataset with all weights
python fiftyone_dataset_creation.py --type carapace --weights_type all

# Create body dataset with all weights  
python fiftyone_dataset_creation.py --type body --weights_type all

# Run error analysis for carapace
python error_flags_analysis_and_filtered_dataset.py --type carapace --weights_type all --error_size mean

# Run error analysis for body
python error_flags_analysis_and_filtered_dataset.py --type body --weights_type all --error_size mean
```

### 4. Launch FiftyOne Visualization
```bash
# For carapace measurements
python ../run_fiftyone_carapace.py

# For body measurements  
python ../run_fiftyone_body.py
```

## Installation and Setup

### Prerequisites
```bash
pip install fiftyone pandas numpy matplotlib seaborn plotly scikit-learn scipy
```

### FiftyOne Setup
```bash
# Install FiftyOne
pip install fiftyone

# Initialize FiftyOne
fiftyone brain init
```

### Directory Setup
```bash
# Create necessary directories
mkdir -p exported_datasets/body_all exported_datasets/carapace_all
mkdir -p fiftyone_datasets
```

## Usage Workflow

### 1. Dataset Creation
```bash
cd Imagej_analysis/

# Create carapace dataset
python fiftyone_dataset_creation.py --type carapace --weights_type all

# Create body dataset
python fiftyone_dataset_creation.py --type body --weights_type all
```

### 2. Error Analysis
```bash
# Run error analysis for carapace measurements
python error_flags_analysis_and_filtered_dataset.py --type carapace --weights_type all --error_size mean

# Run error analysis for body measurements
python error_flags_analysis_and_filtered_dataset.py --type body --weights_type all --error_size mean
```

### 3. Visualization
```bash
# Launch FiftyOne for carapace measurements
python ../run_fiftyone_carapace.py

# Launch FiftyOne for body measurements
python ../run_fiftyone_body.py
```

## Output Files

### Dataset Outputs
- **FiftyOne Datasets**: Interactive datasets with predictions and ground truth
- **Processed Data**: Excel files with calculated real-world measurements
- **Error Analysis**: CSV files with detailed error metrics and flags

### Visualization Outputs
- **Interactive Plots**: Plotly-based interactive visualizations
- **Static Plots**: Matplotlib/Seaborn static plots
- **Statistical Reports**: Detailed error analysis and correlation reports

## Technical Details

### Camera Calibration
The system uses GoPro Hero 11 camera parameters:
- **Resolution**: 5312x2988 pixels
- **FOV**: Configurable horizontal and vertical field of view
- **Distance**: Configurable measurement distance in mm

### Measurement Conversion
Real-world measurements are calculated using:
- **FOV-based scaling**: `scale = (2 * distance * tan(fov/2)) / image_dimension`
- **Angle normalization**: Angles normalized to [0°, 90°] range
- **Combined scaling**: Uses both horizontal and vertical scaling factors

### Error Analysis
The system implements comprehensive error analysis:
- **MAPE Calculation**: Mean Absolute Percentage Error
- **R² Scoring**: Coefficient of determination
- **Correlation Analysis**: Pearson correlation coefficients
- **Multi-flag Detection**: Error categorization by type and severity

### YOLO Keypoint Format
The system processes YOLOv8 keypoint predictions:
- **Format**: `class xc yc w h kp1x kp1y kp1v kp2x kp2y kp2v ...`
- **Keypoints**: start_carapace, eyes, rostrum, tail
- **Normalization**: All coordinates normalized to [0,1] range

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Use random port assignment or specify different ports
2. **Dataset Loading Errors**: Ensure exported datasets exist in correct paths
3. **Memory Issues**: Process datasets in smaller batches
4. **File Path Errors**: Check absolute paths in configuration

### Performance Optimization
- Use SSD storage for large datasets
- Process images in batches
- Adjust FiftyOne memory settings
- Use multiprocessing for large-scale analysis

## Dependencies Summary

- **FiftyOne**: Dataset visualization and management
- **Pandas**: Data manipulation and Excel I/O
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Static visualizations
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Statistical analysis and metrics
- **SciPy**: Statistical functions
- **OpenCV**: Image processing (optional) 