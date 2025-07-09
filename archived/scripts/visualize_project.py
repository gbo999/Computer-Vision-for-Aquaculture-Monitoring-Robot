#!/usr/bin/env python3
"""
Project Structure Visualization Tool

This script generates a visual representation of the project structure,
which can be included in documentation or presentations. It creates
a diagram showing directories and key files.
"""

import os
from pathlib import Path
import argparse
import sys

IGNORE_DIRS = [
    '.git', '__pycache__', '.venv', '.ipynb_checkpoints',
    'runs', 'labels'  # Add project-specific directories to ignore
]

IGNORE_FILES = [
    '.DS_Store', '*.pyc', '*.pyo', '*.pyd', '.Python', '*.so',
    '*.egg', '*.egg-info', '*.png', '*.jpg', '*.jpeg'
]

def should_ignore(path, ignore_dirs, ignore_files):
    """Check if the path should be ignored."""
    path_name = os.path.basename(path)
    
    # Check if it's a directory to ignore
    if os.path.isdir(path) and path_name in ignore_dirs:
        return True
    
    # Check if it's a file pattern to ignore
    if os.path.isfile(path):
        for pattern in ignore_files:
            if pattern.startswith('*'):
                if path_name.endswith(pattern[1:]):
                    return True
            elif pattern == path_name:
                return True
    
    return False

def print_tree(directory, prefix='', ignore_dirs=None, ignore_files=None, max_depth=None, current_depth=0):
    """Print the directory tree."""
    if ignore_dirs is None:
        ignore_dirs = IGNORE_DIRS
    if ignore_files is None:
        ignore_files = IGNORE_FILES
    
    # Check if we've reached max depth
    if max_depth is not None and current_depth > max_depth:
        return
    
    # Get all items in the directory
    items = sorted(os.listdir(directory))
    
    # Filter items based on ignore rules
    items = [item for item in items if not should_ignore(os.path.join(directory, item), ignore_dirs, ignore_files)]
    
    # Process each item
    for i, item in enumerate(items):
        # Check if it's the last item
        is_last = i == len(items) - 1
        
        # Prepare the prefix for the current item
        current_prefix = '└── ' if is_last else '├── '
        
        # Get the full path
        path = os.path.join(directory, item)
        
        # Print the current item
        print(f"{prefix}{current_prefix}{item}")
        
        # If it's a directory, recursively print its contents
        if os.path.isdir(path):
            # Prepare the prefix for the next level
            next_prefix = prefix + ('    ' if is_last else '│   ')
            print_tree(path, next_prefix, ignore_dirs, ignore_files, max_depth, current_depth + 1)

def main():
    """Main function to visualize the project structure."""
    parser = argparse.ArgumentParser(description='Visualize project structure')
    parser.add_argument('--dir', type=str, default='.', help='Directory to visualize')
    parser.add_argument('--depth', type=int, default=None, help='Maximum depth to display')
    parser.add_argument('--output', type=str, default=None, help='Output file (default: stdout)')
    args = parser.parse_args()
    
    # Determine the directory to visualize
    directory = os.path.abspath(args.dir)
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1
    
    # Redirect output if specified
    if args.output:
        sys.stdout = open(args.output, 'w')
    
    # Print the project structure
    print(f"Project Structure: {os.path.basename(directory)}")
    print(".")
    print_tree(directory, max_depth=args.depth)
    
    # Close output file if redirected
    if args.output:
        sys.stdout.close()
        print(f"Project structure saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 