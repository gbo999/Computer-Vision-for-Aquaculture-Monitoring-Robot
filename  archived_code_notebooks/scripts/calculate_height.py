#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate camera distance-height from pixel measurements and real length.

This script calculates the vertical distance from camera to object plane
based on pixel measurements and known real-world length.
"""

import sys
import argparse
import math
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Calculate camera height from pixel measurements and real length.')
    
    parser.add_argument('--image-width', type=int, default=5312,
                        help='Width of the image in pixels (default: 5312)')
    parser.add_argument('--image-height', type=int, default=2988,
                        help='Height of the image in pixels (default: 2988)')
    parser.add_argument('--horizontal-fov', type=float, default=75.2,
                        help='Horizontal field of view in degrees (default: 75.2)')
    parser.add_argument('--vertical-fov', type=float, default=46.0,
                        help='Vertical field of view in degrees (default: 46.0)')
    
    parser.add_argument('--pixel-height', type=float, required=True,
                        help='Height of the object in pixels')
    parser.add_argument('--real-length', type=float, required=True,
                        help='Known real length of the object in millimeters')
    
    return parser.parse_args()


def calculate_distance_height(image_height, vertical_fov, pixel_height, real_length_mm):
    """
    Calculate the camera distance-height based on known object length.
    
    Args:
        image_height (int): Height of the image in pixels
        vertical_fov (float): Vertical field of view in degrees
        pixel_height (float): Height of the object in pixels
        real_length_mm (float): Known real length of the object in millimeters
        
    Returns:
        float: Distance from camera to object plane in millimeters
    """
    # First, calculate the scaling factor (mm per pixel)
    scale_y = real_length_mm / pixel_height
    
    # Convert FOV to radians
    vertical_fov_rad = math.radians(vertical_fov)
    
    # Rearrange the equation from ObjectLengthMeasurer.calculate_scaling_factors():
    # scale_y = (2 * distance_mm * math.tan(fov_y_rad / 2)) / image_height
    # to solve for distance_mm:
    distance_mm = (scale_y * image_height) / (2 * math.tan(vertical_fov_rad / 2))
    
    return distance_mm


def main():
    """Calculate camera height from object based on real length and pixel height."""
    args = parse_arguments()
    
    # Calculate the camera distance-height
    camera_height_mm = calculate_distance_height(
        args.image_height,
        args.vertical_fov,
        args.pixel_height,
        args.real_length
    )
    
    # Print results
    print(f"Image dimensions: {args.image_width}x{args.image_height} pixels")
    print(f"Camera FOV: {args.horizontal_fov}°×{args.vertical_fov}°")
    print(f"Object measurements: {args.pixel_height} pixels = {args.real_length} mm")
    print(f"CALCULATED CAMERA HEIGHT: {camera_height_mm:.2f} mm")
    

if __name__ == "__main__":
    main() 