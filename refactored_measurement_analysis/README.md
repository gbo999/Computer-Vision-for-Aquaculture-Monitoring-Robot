# Refactored Prawn Measurement Analysis

A modular, maintainable system for analyzing prawn measurements using computer vision and pose estimation techniques with FiftyOne visualization.

## Overview

This package refactors the original monolithic measurement analysis code into a clean, modular architecture that separates concerns and provides better maintainability, testability, and extensibility.

### Key Improvements

- **Modular Design**: Separated data loading, processing, measurement calculation, and visualization
- **Unified Workflows**: Combined carapace and body measurement pipelines
- **Clean Architecture**: Each component has a single responsibility
- **Better Error Handling**: Comprehensive error handling and progress reporting
- **Professional CLI**: Clean command-line interface with validation
- **Type Safety**: Full type hints throughout the codebase
- **Documentation**: Comprehensive documentation and examples

## Architecture

```
refactored_measurement_analysis/
├── __init__.py              # Package initialization
├── config.py                # Configuration management
├── models.py                # Data models and calculations
├── data_processing.py       # File I/O and data processing
├── visualization.py         # FiftyOne visualization
├── measurement_analysis.py  # Core analysis orchestration
├── main.py                  # Command-line interface
└── README.md               # This file
```

### Components

#### 1. Configuration (`config.py`)
- **CameraConfig**: Camera parameters (FOV, focal length, image dimensions)
- **MeasurementConfig**: Measurement type configurations (keypoints, skeleton)
- **Config**: Main configuration class with paths and settings

#### 2. Data Models (`models.py`)
- **PoseDetection**: YOLO pose detection parsing
- **ObjectLengthMeasurer**: FOV-based measurement calculations
- **FocalLengthMeasurer**: Alternative focal length method
- **PrawnMeasurements**: Complete measurement data with error calculations

#### 3. Data Processing (`data_processing.py`)
- **DataLoader**: Unified data loading for carapace/body measurements
- **YOLOParser**: YOLO format file parsing
- **FilenameProcessor**: Filename cleaning and ID extraction
- **FilePathResolver**: File path resolution across directory structures

#### 4. Visualization (`visualization.py`)
- **FiftyOneDatasetManager**: Dataset creation and persistence
- **FiftyOneSampleProcessor**: Sample processing for visualization
- **VisualizationCreator**: Polyline and annotation creation
- **ErrorTagManager**: Error-based sample tagging

#### 5. Analysis Orchestration (`measurement_analysis.py`)
- **MeasurementAnalyzer**: Main analysis workflow coordination
- Integrates all components for complete analysis pipeline
- Handles detection matching, measurement calculation, and result generation

#### 6. Command-Line Interface (`main.py`)
- Professional argument parsing with validation
- Support for different measurement types and weight configurations
- Interactive FiftyOne app launching

## Usage

### Command Line Interface

```bash
# Basic usage
python -m refactored_measurement_analysis.main --type carapace --weights car

# With custom port and verbose output
python -m refactored_measurement_analysis.main --type body --weights all --port 5160 --verbose

# Save results to specific directory without FiftyOne
python -m refactored_measurement_analysis.main --type carapace --weights kalkar --output-dir ./results --no-fiftyone
```

### Programmatic Usage

```python
from refactored_measurement_analysis import MeasurementAnalyzer

# Initialize analyzer
analyzer = MeasurementAnalyzer(
    measurement_type='carapace',
    weights_type='car',
    verbose=True
)

# Run analysis
results_df, dataset_name = analyzer.run_analysis(
    output_dir=Path('./results'),
    create_fiftyone=True
)

# Get summary statistics
summary = analyzer.get_analysis_summary()
print(f"Mean error: {summary['mean_error_percentage']:.2f}%")

# Launch FiftyOne app
app = analyzer.launch_fiftyone_app()
```

### Configuration Customization

```python
from refactored_measurement_analysis.config import Config

# Access configuration
config = Config()
print(f"Camera FOV: {config.CAMERA.horizontal_fov}°")
print(f"Image dimensions: {config.CAMERA.image_width}x{config.CAMERA.image_height}")

# Modify paths if needed
config.BASE_PATH = Path("/custom/path")
```

## Measurement Types

### Carapace Measurements
- **Keypoints**: start-carapace → eyes
- **Use Case**: Carapace length measurement
- **Data Source**: CSV file with Length_1, Length_2, Length_3 columns

### Body Measurements  
- **Keypoints**: tail → rostrum
- **Use Case**: Full body length measurement
- **Data Source**: Excel file with body_length_1, body_length_2, body_length_3 columns

## Weight Types

- **car**: Trained on car pond data
- **kalkar**: Trained on kalkar pond data
- **all**: Trained on combined dataset

## Output Files

### Excel Results
Comprehensive analysis results with columns:
- Basic measurements (predicted_length_mm, manual_min_length, etc.)
- Error metrics (mpe_min, mpe_max, mpe_median, etc.)
- Angle and scale information
- Ground truth comparisons (if available)

### FiftyOne Dataset
Interactive visualization with:
- Pose detection overlays
- Measurement line annotations
- Error-based sample tagging
- Filterable metadata fields

## Error Analysis

The system calculates multiple error metrics:
- **MPE (Mean Percentage Error)**: Against min, max, median manual measurements
- **Absolute Error**: Raw difference in millimeters
- **Ground Truth Comparison**: If ground truth annotations available
- **Focal Length Comparison**: Alternative calculation method

### Error Tags
Samples are automatically tagged based on error thresholds:
- `excellent`: < 5% error
- `very_low_error`: 5-10% error
- `low_error`: 10-25% error
- `medium_error`: 25-50% error
- `high_error`: > 50% error

## Migration from Original Code

### Key Changes

1. **Separated Concerns**: 
   - Original: Mixed data loading, processing, and visualization in single functions
   - Refactored: Each component has a single responsibility

2. **Eliminated Duplication**:
   - Original: Separate carapace/body processing with duplicated code
   - Refactored: Unified workflow with configuration-driven differences

3. **Improved Error Handling**:
   - Original: Basic error handling with limited feedback
   - Refactored: Comprehensive error handling with progress reporting

4. **Better Configuration**:
   - Original: Hardcoded paths scattered throughout
   - Refactored: Centralized configuration management

### Migration Steps

1. **Update Imports**:
   ```python
   # Old
   from measurements_analysis import run_analysis
   
   # New  
   from refactored_measurement_analysis import MeasurementAnalyzer
   ```

2. **Update Function Calls**:
   ```python
   # Old
   results = run_analysis(measurement_type, weights_type)
   
   # New
   analyzer = MeasurementAnalyzer(measurement_type, weights_type)
   results_df, dataset_name = analyzer.run_analysis()
   ```

3. **Update Configuration**:
   - Review `config.py` and update paths as needed
   - Ensure data files are in expected locations

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computations  
- fiftyone: Visualization and dataset management
- tqdm: Progress bars
- pathlib: Path handling
- dataclasses: Data structures
- typing: Type hints

## Future Extensions

The modular architecture makes it easy to extend:

1. **New Measurement Types**: Add to `MEASUREMENT_CONFIGS` in `config.py`
2. **Alternative Calculators**: Implement new measurement methods in `models.py`
3. **Additional Visualizations**: Extend `VisualizationCreator` class
4. **Custom Error Metrics**: Add methods to `PrawnMeasurements.calculate_errors()`
5. **New Data Sources**: Extend `DataLoader` for different file formats

## Performance

- **Parallel Processing**: Ready for parallel processing extensions
- **Memory Efficient**: Processes data in batches
- **Caching**: Configuration and file resolution caching
- **Progress Tracking**: Real-time progress reporting

## Testing

The modular design enables comprehensive testing:
- Unit tests for individual components
- Integration tests for workflows
- Mock data for testing edge cases
- Configuration validation tests

---

This refactored system provides a solid foundation for prawn measurement analysis with room for future enhancements and easy maintenance. 