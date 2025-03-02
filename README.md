# Counting Research Algorithms for Aquaculture Monitoring

## Overview

This repository contains algorithms and tools developed for automated counting and measurement of aquatic organisms (specifically crayfish exuviae) for aquaculture monitoring systems. The project implements computer vision and machine learning techniques to automate previously manual monitoring processes.

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
from src.counting import detector
from src.measurement import size_analyzer

# Load and process an image
results = detector.detect_exuviae("path/to/image.jpg")
measurements = size_analyzer.measure_all(results)

# Analyze the results
print(f"Found {len(results)} exuviae with average size {measurements.mean()}")
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


