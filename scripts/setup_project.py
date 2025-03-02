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
    },
    "notebooks": {},
    "src": {
        "counting": {},
        "measurement": {},
        "data": {},
        "utils": {},
    },
    "scripts": {},
    "tests": {},
    "docs": {
        "workflow": {},
        "api": {},
        "usage": {},
        "images": {},
    },
    "models": {},
    "results": {
        "figures": {},
        "reports": {},
    },
    ".github": {
        "workflows": {},
    },
}

# Files to move to their appropriate locations
FILES_TO_MOVE = {
    "segment_molt.py": "scripts/",
    "visualize_predictions.py": "scripts/",
    "show_examples.py": "scripts/",
    "analyze_sizes_by_group.py": "scripts/",
    "analyze_sizes.py": "scripts/",
    "filter_predictions.py": "scripts/",
    "pred.py": "scripts/",
    "organize_test_data.py": "scripts/",
    "size_analysis.png": "results/figures/",
    "exuviae_analysis_workflow.txt": "docs/workflow/original_workflow.txt",
}

# Sample README files to create
README_TEMPLATES = {
    "data/README.md": """# Data Directory

This directory contains data used in the counting research algorithms project.

## Structure

- `raw/`: Raw data collected from experiments
- `processed/`: Data after preprocessing steps

## Data Sources

[Describe your data sources and collection methods]

""",
    "notebooks/README.md": """# Analysis Notebooks

This directory contains Jupyter notebooks used for exploratory data analysis, 
algorithm development, and results visualization.

## Notebooks

- [List of notebooks and their purposes]

""",
    "tests/README.md": """# Tests

This directory contains unit and integration tests for the counting algorithms.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_detection.py
```

""",
    "models/README.md": """# Model Directory

This directory contains model definition files and saved model weights.

## Models

- [List of models and their purposes]
- [Performance metrics]

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
            print(f"Warning: Source file {source_path} does not exist. Skipping.")
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
        
        # Move the file
        if os.path.exists(source_path):
            print(f"Moving {source_path} to {target_file}")
            # Use copy instead of move to be safe
            shutil.copy2(source_path, target_file)
            # Optionally uncomment to remove the original
            # os.remove(source_path)


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
    
    print("\nProject setup complete!")
    print("\nNext steps:")
    print("1. Review the new directory structure")
    print("2. Update the README.md with your project-specific information")
    print("3. Commit the changes to your repository")


if __name__ == "__main__":
    main() 