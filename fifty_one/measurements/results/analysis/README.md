# Refactored Prawn Measurement Analysis

This directory contains a completely refactored version of the prawn measurement analysis system, designed with modular architecture and clean separation of concerns.

## 🚀 Key Improvements

### From Original Code
- **measurements_analysis.py**: 190 lines → **main.py**: Clean CLI interface
- **data_loader.py**: 1646 lines → Separated into 5 focused modules
- **Eliminated**: Hardcoded paths, duplicate code, mixed responsibilities
- **Added**: Configuration management, better error handling, extensible design

### Architecture Benefits
- **Maintainable**: Clear separation of concerns across modules
- **Testable**: Focused functions with single responsibilities  
- **Configurable**: Centralized configuration management
- **Extensible**: Easy to add new measurement types or features
- **Robust**: Better error handling and validation

## 📁 Module Structure

```
fifty_one/measurements/results/analysis/
├── main.py                    # 🎯 Main entry point with CLI
├── config.py                  # ⚙️  Configuration and constants
├── models.py                  # 📊 Data models and measurement classes
├── data_processing.py         # 🔄 Data loading and file processing
├── measurement_analysis.py    # 🧮 Core analysis orchestration
├── visualization.py           # 👁️  FiftyOne visualization management
└── README.md                  # 📖 This documentation
```

## 🎯 Quick Start

### Basic Usage
```bash
# Run body measurement analysis (default)
python main.py

# Run carapace measurement analysis with specific weights
python main.py --type carapace --weights car

# Use custom port for visualization
python main.py --type body --weights all --port 5160

# Enable verbose logging
python main.py --type carapace --verbose
```

### Advanced Usage
```bash
# Custom output directory
python main.py --output-dir /path/to/custom/output

# Skip processing if dataset exists
python main.py --skip-existing

# Use custom configuration
python main.py --config custom_config.py
```

## 📊 Module Details

### 1. `config.py` - Configuration Management
**What it does**: Centralizes all hardcoded paths, camera parameters, and system configuration.

**Key improvements from original**:
- Eliminated scattered hardcoded paths throughout codebase
- Used `pathlib.Path` for robust path handling
- Added configurable camera parameters and thresholds
- Provided helper methods for common path operations

```python
# Example usage
config = Config()
prediction_path = config.get_prediction_path('all')
dataset_name = config.get_dataset_name('carapace', 'car')
```

### 2. `models.py` - Data Models and Calculations
**What it does**: Defines data structures and measurement calculation classes.

**Key classes**:
- `PoseDetection`: Represents YOLO pose detections
- `ObjectLengthMeasurer`: FOV-based measurement calculations  
- `FocalLengthMeasurer`: Focal length-based calculations
- `PrawnMeasurements`: Complete measurement data structure

**Improvements from original**:
- Extracted `ObjectLengthMeasurer` class from 200+ line functions
- Added clear documentation for measurement algorithms
- Unified error calculation methods
- Type hints for better code clarity

### 3. `data_processing.py` - Data Loading and Processing
**What it does**: Handles file I/O, data parsing, and preprocessing.

**Key classes**:
- `DataLoader`: Unified data loading for carapace/body measurements
- `YOLOParser`: Parse YOLO format pose files
- `FilenameProcessor`: Handle various filename formats and identifier extraction
- `BoundingBoxProcessor`: Bounding box calculations and conversions
- `PrawnDataProcessor`: Extract prawn measurements from CSV data
- `FilePathResolver`: Resolve prediction and ground truth file paths

**Improvements from original**:
- Eliminated duplicate filename processing code
- Better error handling for file operations
- Centralized bounding box calculations
- Clear separation between different data processing tasks

### 4. `measurement_analysis.py` - Core Analysis Engine
**What it does**: Orchestrates the complete analysis workflow.

**Key classes**:
- `MeasurementAnalyzer`: Main analysis coordinator
- `PortManager`: Manages FiftyOne port allocation

**Improvements from original**:
- Clean separation between analysis logic and visualization
- Better error handling and progress reporting
- Configurable analysis parameters
- Unified processing for different measurement types

### 5. `visualization.py` - FiftyOne Visualization
**What it does**: Manages FiftyOne dataset creation, sample processing, and visualization.

**Key classes**:
- `FiftyOneDatasetManager`: Dataset creation and persistence
- `FiftyOneSampleProcessor`: Individual sample processing
- `VisualizationCreator`: Creates polylines and visual annotations
- `ErrorTagManager`: Tags samples based on error thresholds

**Improvements from original**:
- Eliminated 500+ line functions with mixed responsibilities
- Separated dataset management from visualization logic
- Configurable visualization parameters
- Better error-based tagging system

### 6. `main.py` - Clean CLI Interface
**What it does**: Provides clean command-line interface with proper argument parsing and validation.

**Improvements from original**:
- Professional CLI with help documentation
- Input validation and error handling
- Flexible parameter options
- Better user feedback and progress reporting

## 🔧 Configuration

The system uses a centralized configuration approach. Key configuration areas:

### Camera Parameters
```python
CAMERA = CameraConfig(
    image_width=5312,
    image_height=2988,
    horizontal_fov=76.2,
    vertical_fov=46.0,
    focal_length_default=24.72,
    focal_length_left_right=23.64,
    pixel_size=0.00716844
)
```

### File Paths
All paths are configurable and use `pathlib.Path` for robustness:
```python
CARAPACE_DATA_PATH = BASE_PATH / "src/measurement/ImageJ/Filtered_Data.csv"
BODY_DATA_PATH = BASE_PATH / "src/measurement/ImageJ/final_full_statistics_with_prawn_ids_and_uncertainty - Copy.xlsx"
```

### Error Thresholds
```python
ERROR_THRESHOLDS = {
    'high': 50,     # >50% error
    'medium': 25,   # 25-50% error  
    'low': 10,      # 10-25% error
    'very_low': 5   # 5-10% error
}
```

## 🧪 Key Technical Improvements

### 1. Measurement Calculation Refactoring
**Original**: 200+ line functions mixing calculation, file I/O, and visualization
**Refactored**: Clean `ObjectLengthMeasurer` class with focused methods

```python
# Original approach (simplified)
def process_detection(closest_detection, sample, filename, prawn_id, filtered_df, ground):
    # 200+ lines mixing calculations, file operations, dataframe updates
    
# Refactored approach  
measurer = ObjectLengthMeasurer(width, height, h_fov, v_fov, distance)
result = measurer.compute_length_between_points(point1, point2)
errors = prawn_measurement.calculate_errors()
```

### 2. Data Processing Separation
**Original**: Mixed data loading, filename processing, and bounding box calculations
**Refactored**: Dedicated classes for each responsibility

```python
# Original: Everything mixed together
def add_metadata(sample, filename, filtered_df, metadata_df, swimmingdf=None):
    # Complex filename processing mixed with metadata lookup
    
# Refactored: Clear separation
filename_processor = FilenameProcessor()
identifier = filename_processor.extract_identifier(filename)
compatible_name = filename_processor.create_compatible_filename(filename)
```

### 3. Visualization Logic Cleanup
**Original**: 100+ line functions creating polylines mixed with data processing
**Refactored**: Dedicated `VisualizationCreator` class

```python
# Original: Mixed concerns
def add_prawn_detections(sample, matching_rows, filtered_df, filename):
    # 150+ lines mixing data extraction and visualization creation
    
# Refactored: Clean separation
viz_creator = VisualizationCreator(config)
viz_creator.add_measurement_visualizations(sample, prawn_measurements, measurement_type)
```

## 📈 Usage Examples

### Running Different Analysis Types
```bash
# Carapace measurements with car weights
python main.py --type carapace --weights car

# Body measurements with all weights 
python main.py --type body --weights all

# Carapace with kalkar weights on custom port
python main.py --type carapace --weights kalkar --port 5160
```

### Custom Configuration
```python
# Example of extending configuration
class CustomConfig(Config):
    # Override specific paths
    PREDICTION_PATHS = {
        'custom': Path('/path/to/custom/predictions'),
        **Config.PREDICTION_PATHS
    }
```

## 🔍 Error Handling

The refactored system includes comprehensive error handling:

- **File validation**: Checks all required files exist before processing
- **Port management**: Automatically finds available ports
- **Input validation**: Validates all CLI arguments
- **Graceful degradation**: Continues processing when possible
- **Detailed logging**: Optional verbose mode for debugging

## 🚀 Future Extensions

The modular architecture makes it easy to add:

- **New measurement types**: Add to `MEASUREMENT_CONFIGS`
- **Additional visualizations**: Extend `VisualizationCreator`
- **Different data sources**: Add loaders to `DataLoader`
- **Custom analysis methods**: Extend `MeasurementAnalyzer`
- **Alternative backends**: Replace FiftyOne components as needed

## 📊 Performance Improvements

- **Reduced memory usage**: No longer loads entire files into memory unnecessarily
- **Better error handling**: Fails fast on invalid inputs
- **Parallel processing ready**: Architecture supports future parallelization
- **Caching opportunities**: Configuration and data loading can be cached

## 🔄 Migration Guide

To migrate from the original system:

1. **Replace** `python measurements_analysis.py` → `python main.py`
2. **Update** any hardcoded paths in your scripts to use the new configuration
3. **Modify** any custom extensions to use the new modular classes
4. **Test** with the same parameters to ensure equivalent results

The refactored system maintains full compatibility with the original analysis results while providing a much cleaner and more maintainable codebase. 