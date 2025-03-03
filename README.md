# Counting Research Algorithms for Aquaculture Monitoring

## Overview

This repository contains specialized algorithms and tools developed for automated counting and measurement of aquatic organisms (specifically crayfish exuviae) for aquaculture monitoring systems. The project implements computer vision and machine learning techniques to automate previously manual monitoring processes.

## Core Components

The repository is organized around two primary algorithmic pillars:

### 1. Counting Algorithms
Detection and counting of organisms in underwater imagery:
- Object detection using YOLOv8 architecture
- Specialized filtering to remove false positives
- Tracking across video frames for consistent counts

### 2. Measurement Algorithms
Accurate size determination of detected organisms:
- Image segmentation to isolate individual organisms
- Keypoint detection for dimensional analysis
- Calibrated measurement with distortion correction
- Statistical analysis of size distributions

These components work together in an integrated pipeline that processes raw imagery and produces quantitative data for ecological monitoring.

## Features

- **Automated Detection**: YOLOv8-based detection of crayfish exuviae in underwater imagery
- **Image Segmentation**: Color-based segmentation to highlight exuviae against complex backgrounds
- **Size Measurement**: Automated measurement algorithms for detected organisms
- **Analysis Pipeline**: Complete workflow from raw imagery to ecological analysis
- **Visualization Tools**: Data visualization for monitoring and research insights

## Research Context

This work was developed as part of research into improving aquaculture monitoring systems. The algorithms enable:
- Non-invasive monitoring of crayfish populations
- Automated molt detection for growth tracking
- Quantitative analysis of population health indicators

## Project Architecture

```
counting_research_algorithms/
├── src/
│   ├── counting/      # Detection and counting algorithms
│   ├── measurement/   # Size measurement algorithms
│   ├── data/          # Data processing utilities
│   └── utils/         # Shared utility functions
├── scripts/           # Analysis and processing scripts
├── notebooks/         # Research and analysis notebooks
└── ...                # Documentation, tests, etc.
```

## Workflow

The pipeline consists of several stages:
1. **Image Collection** - Gathering underwater imagery
2. **Preprocessing** - Enhancing images for better detection
3. **Object Detection** - Identifying organisms in images
4. **Measurement** - Determining size of detected organisms
5. **Analysis** - Statistical processing of measurements

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/counting_research_algorithms.git
cd counting_research_algorithms

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage Example

```python
# Example combining counting and measurement
from src.counting.detection import detect_exuviae
from src.measurement.metrics import calculate_dimensions

# Detect exuviae in image
detections = detect_exuviae("path/to/image.jpg")

# Measure each detected object
measurements = []
for detection in detections:
    dimensions = calculate_dimensions(detection)
    measurements.append(dimensions)

# Analyze the results
print(f"Found {len(measurements)} exuviae")
print(f"Average length: {sum(m['length'] for m in measurements) / len(measurements):.2f} mm")
```

## Documentation

For detailed documentation on the workflow and algorithms, see the [docs](./docs) directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
[Citation information will be added when published]
```

## Contributors

- Your Name - [Your Institution]

## Acknowledgments

- Acknowledgment to collaborators and funding sources


