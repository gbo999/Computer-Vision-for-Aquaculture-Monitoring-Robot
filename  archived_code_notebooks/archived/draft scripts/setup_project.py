#!/usr/bin/env python3
"""
Project Setup Script for Counting Research Algorithms

This script organizes the repository structure according to best practices
for scientific research code. It creates necessary directories, moves files
to appropriate locations, and ensures a consistent project structure.
"""

import os
import shutil
from pathlib import Path
import sys

# Define the project structure
PROJECT_STRUCTURE = {
    "data": {
        "raw": {},
        "processed": {},
        "test_sets": {
            "test-left": {},
            "test-right": {},
            "test-car": {},
        },
    },
    "src": {
        "counting": {
            "detection": {},
            "tracking": {},
            "validation": {},
        },
        "measurement": {
            "segmentation": {},
            "keypoints": {},
            "calibration": {},
            "image_enhancement": {},
        },
        "data": {
            "loaders": {},
            "augmentation": {},
            "preparation": {},
        },
        "utils": {
            "visualization": {},
            "metrics": {},
            "io": {},
        },
    },
    "notebooks": {
        "exploration": {},
        "prototyping": {},
        "analysis": {},
    },
    "scripts": {
        "preprocessing": {},
        "analysis": {},
        "visualization": {},
        "deployment": {},
        "tools": {},
    },
    "tests": {
        "counting": {},
        "measurement": {},
        "utils": {},
    },
    "models": {
        "detection": {},
        "segmentation": {},
    },
    "results": {
        "figures": {},
        "reports": {},
    },
    "docs": {
        "workflow": {},
        "api": {},
        "usage": {},
        "images": {},
    },
    "docker": {},
    ".github": {
        "workflows": {},
    },
}

# Files to move to their appropriate locations
FILES_TO_MOVE = {
    # Python scripts
    "segment_molt.py": "scripts/preprocessing/",
    "organize_test_data.py": "scripts/preprocessing/",
    "analyze_sizes.py": "scripts/analysis/",
    "analyze_sizes_by_group.py": "scripts/analysis/",
    "filter_predictions.py": "scripts/analysis/",
    "visualize_predictions.py": "scripts/visualization/",
    "show_examples.py": "scripts/visualization/",
    "pred.py": "scripts/deployment/",
    
    # Result files
    "size_analysis.png": "results/figures/",
    
    # Documentation
    "exuviae_analysis_workflow.txt": "docs/workflow/original_workflow.txt",
    
    # Configuration files
    "Dockerfile": "docker/",
    
    # Test data directories
    "test-left": "data/test_sets/",
    "test-right": "data/test_sets/",
    "test-car": "data/test_sets/",
    
    # Calibration files
    "standard_calibration.json": "src/measurement/calibration/",
    
    # Move notebook directories
    "colab_notebooks": "notebooks/",
    "cpu_predictions _notebooks": "notebooks/",
    
    # Other files that need better organization
    "f.py": "scripts/tools/legacy_f.py",  # Generic filenames should be renamed meaningfully
    "g.py": "scripts/tools/legacy_g.py",
}

# Sample README files to create
README_TEMPLATES = {
    "data/README.md": """# Data Directory

This directory contains data used in the counting research algorithms project.

## Structure

- `raw/`: Raw data collected from experiments
- `processed/`: Data after preprocessing steps
- `test_sets/`: Organized test datasets

## Data Sources

[Describe your data sources and collection methods]

""",
    "notebooks/README.md": """# Analysis Notebooks

This directory contains Jupyter notebooks used for exploratory data analysis, 
algorithm development, and results visualization.

## Notebooks

- `exploration/`: Data exploration notebooks
- `prototyping/`: Algorithm prototyping and development
- `analysis/`: Results analysis and visualization

""",
    "tests/README.md": """# Tests

This directory contains unit and integration tests for the counting and measurement algorithms.

## Structure

- `counting/`: Tests for counting modules
- `measurement/`: Tests for measurement modules
- `utils/`: Tests for utility functions

## Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/counting/
```

""",
    "models/README.md": """# Model Directory

This directory contains model definition files and saved model weights.

## Models

- `detection/`: Object detection models (e.g., YOLOv8)
- `segmentation/`: Segmentation models

## Performance Metrics

[Include performance metrics for each model]

""",
    "src/README.md": """# Source Code Directory

This directory contains the core functionality of the counting research algorithms.

## Structure

- `counting/`: Algorithms for detecting and counting objects
- `measurement/`: Algorithms for measuring detected objects
- `data/`: Data processing utilities
- `utils/`: General utility functions

## Usage

See the main README for usage examples.

""",
}


def create_directory_structure(base_path, structure, level=0):
    """Create the directory structure recursively."""
    for directory, subdirs in structure.items():
        dir_path = os.path.join(base_path, directory)
        
        # Create directory if it doesn't exist
        if not os.path.exists(dir_path):
            print(f"{'  ' * level}Creating directory: {dir_path}")
            os.makedirs(dir_path)
        
        # Recursively create subdirectories
        if subdirs:
            create_directory_structure(dir_path, subdirs, level + 1)


def move_files(base_path, files_map):
    """Move files to their appropriate locations."""
    for source_file, target_dir in files_map.items():
        source_path = os.path.join(base_path, source_file)
        target_path = os.path.join(base_path, target_dir)
        
        # Skip if source file doesn't exist
        if not os.path.exists(source_path):
            print(f"Warning: Source file/directory {source_path} does not exist. Skipping.")
            continue
        
        # Create target directory if it doesn't exist
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        
        # Determine target file path
        if target_dir.endswith('/'):
            # Keep the same filename
            target_file = os.path.join(target_path, os.path.basename(source_path))
        else:
            # Use the specified filename
            target_file = target_path
        
        # Move the file or directory
        if os.path.exists(source_path):
            print(f"Moving {source_path} to {target_file}")
            # Use copy instead of move to be safe
            if os.path.isdir(source_path):
                if os.path.exists(target_file):
                    shutil.rmtree(target_file, ignore_errors=True)
                shutil.copytree(source_path, target_file)
            else:
                shutil.copy2(source_path, target_file)
            # Optionally uncomment to remove the original
            # if os.path.isdir(source_path):
            #     shutil.rmtree(source_path)
            # else:
            #     os.remove(source_path)


def create_readme_files(base_path, readme_templates):
    """Create README files from templates."""
    for file_path, content in readme_templates.items():
        full_path = os.path.join(base_path, file_path)
        
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(full_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # Create the README file if it doesn't exist
        if not os.path.exists(full_path):
            print(f"Creating README: {full_path}")
            with open(full_path, 'w') as f:
                f.write(content)


def create_init_files(base_path, directory="src"):
    """Create __init__.py files in all subdirectories of the given directory."""
    for root, dirs, files in os.walk(os.path.join(base_path, directory)):
        init_file = os.path.join(root, "__init__.py")
        if not os.path.exists(init_file):
            print(f"Creating __init__.py in {root}")
            with open(init_file, 'w') as f:
                f.write("# Package initialization\n")


def main():
    """Main function to set up the project structure."""
    # Get the base directory (repository root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"Setting up project structure in: {base_dir}")
    
    # 1. Create the directory structure
    create_directory_structure(base_dir, PROJECT_STRUCTURE)
    
    # 2. Move files to their appropriate locations
    move_files(base_dir, FILES_TO_MOVE)
    
    # 3. Create README files
    create_readme_files(base_dir, README_TEMPLATES)
    
    # 4. Create __init__.py files
    create_init_files(base_dir)
    
    print("\nProject setup complete!")
    print("\nNext steps:")
    print("1. Review the new directory structure")
    print("2. Update README.md files with project-specific information")
    print("3. Run tests to ensure everything works correctly")
    print("4. Commit the changes to your repository")


if __name__ == "__main__":
    main() 