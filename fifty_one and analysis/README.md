# Image Directories Used in FiftyOne Analysis

This document lists the image directories that are referenced and used in the FiftyOne analysis code.

## Main Image Directories

### 1. imagej Measurements
Base path: `/measurements/carapace/`
- `right/` - Female prawn images
- `left/` - Male prawn images
- `car/` - Square pond images

Used in: `measurements_analysis.py` for keypoint detection and measurement analysis

### 2. Molt/Exuviae Analysis
Base path: `/measurement_paper_images/molt/all molt/undistorted/resized/`
- Contains undistorted and resized molt images
- `segmented/` subdirectory contains colorized versions

Used in: `binary_exuviae_colorizer.py` and `exuviae_measurement_analyzer.py`

### 3. Drone Detection
Base path: `/measurement_paper_images/detection drone/`
- Contains aerial pond images
- `runs-detections-drone-14.08/` contains model outputs

Used in: `counting.ipynb` for prawn counting analysis

## Note on Data Storage
These image directories are stored in OneDrive and referenced by absolute paths in the code. When running the analysis:

1. Make sure you have access to the OneDrive folders
2. Update the paths in the code to match your OneDrive locations
3. The images are not included in the git repository due to their size 