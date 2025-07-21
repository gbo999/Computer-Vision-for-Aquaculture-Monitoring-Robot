#!/usr/bin/env python3
"""
Image Colorizer - Apply colorize effect to masked regions of an image
"""

import numpy as np
import cv2
import argparse
import os

def blend2(left, right, pos):
    """
    Linear interpolation between two RGB colors.
    
    Args:
        left: RGB tuple or array (left color)
        right: RGB tuple or array (right color)
        pos: float between 0 and 1 (blend position)
    
    Returns:
        Blended RGB as numpy array
    """
    return (1 - pos) * np.array(left) + pos * np.array(right)

def blend3(left, main, right, pos):
    """
    Three-way blend between colors based on position.
    
    Args:
        left: RGB tuple or array (left color)
        main: RGB tuple or array (middle color)
        right: RGB tuple or array (right color)
        pos: float from -1 to 1 (blend position)
    
    Returns:
        Blended RGB as numpy array
    """
    if pos < 0:
        return blend2(left, main, pos + 1)
    elif pos > 0:
        return blend2(main, right, pos)
    else:
        return np.array(main)

def colorize(pixel_value, hue_rgb, saturation, lightness):
    """
    Colorize a pixel based on hue, saturation, and lightness.
    
    Args:
        pixel_value: float between 0 and 1 (original pixel's HSV value)
        hue_rgb: tuple/array of RGB values for the target hue (at full saturation and value)
        saturation: float between 0 and 1 (amount of color to apply)
        lightness: float between -1 and 1 (lightness adjustment)
    
    Returns:
        Colorized RGB value as numpy array
    """
    # Convert inputs to numpy arrays for easier calculation
    hue_rgb = np.array(hue_rgb)
    gray = np.array([128, 128, 128])
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])
    
    # Create the base colorized value
    color = blend2(gray, hue_rgb, saturation)
    
    # Apply lightness adjustments
    if lightness <= -1:
        return black
    elif lightness >= 1:
        return white
    elif lightness >= 0:
        pos = 2 * (1 - lightness) * (pixel_value - 1) + 1
        return blend3(black, color, white, pos)
    else:
        pos = 2 * (1 + lightness) * pixel_value - 1
        return blend3(black, color, white, pos)

def colorize_masked_regions(image, mask, hue_rgb, saturation, lightness):
    """
    Applies colorize effect only to regions where mask == 255
    
    Args:
        image: RGB or BGR image as numpy array
        mask: Single-channel mask (0 or 255)
        hue_rgb: RGB color for target hue
        saturation: float 0-1 representing saturation level
        lightness: float -1 to 1 representing lightness adjustment
    
    Returns:
        Image with colorize effect applied to masked regions
    """
    # Create a copy of the input image to avoid modifying the original
    result = image.copy()
    
    # Find pixels where mask is 255
    y_coords, x_coords = np.where(mask == 255)
    
    # Convert image to HSV to extract value channel
    if len(image.shape) == 3 and image.shape[2] == 3:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image.dtype == np.uint8 else cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    else:
        raise ValueError("Image must be a 3-channel color image")
    
    # Process each pixel in the masked region
    for y, x in zip(y_coords, x_coords):
        # Get original pixel value (V in HSV)
        pixel_value = hsv_image[y, x, 2] / 255.0  # Normalize to 0-1
        
        # Apply colorize function
        new_color = colorize(pixel_value, hue_rgb, saturation, lightness)
        
        # Ensure result is within valid range
        new_color = np.clip(new_color, 0, 255).astype(result.dtype)
        
        # Update the pixel in the result image
        result[y, x] = new_color

    return result

def colorize_masked_regions_vectorized(image, mask, hue_rgb, saturation, lightness):
    """
    Vectorized version for better performance with large images
    
    Args:
        image: RGB or BGR image as numpy array
        mask: Single-channel mask (0 or 255)
        hue_rgb: RGB color for target hue
        saturation: float 0-1 representing saturation level
        lightness: float -1 to 1 representing lightness adjustment
    
    Returns:
        Image with colorize effect applied to masked regions
    """
    # Create a copy of the input image
    result = image.copy()
    
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image.dtype == np.uint8 else cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    
    # Extract pixels where mask == 255
    mask_bool = mask == 255
    if not np.any(mask_bool):
        print("Warning: No pixels in mask with value 255")
        return result
    
    # Get pixel values from masked regions
    pixel_values = hsv_image[mask_bool, 2] / 255.0
    
    # Create gray, black, white arrays for blending
    gray = np.array([128, 128, 128])
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])
    
    # Base color
    color = blend2(gray, hue_rgb, saturation)
    
    # Initialize array for colorized values
    new_colors = np.zeros((np.sum(mask_bool), 3), dtype=np.float32)
    
    # Handle extreme lightness cases
    if lightness <= -1:
        new_colors[:] = black
    elif lightness >= 1:
        new_colors[:] = white
    else:
        # Calculate positions based on lightness value
        if lightness >= 0:
            pos = 2 * (1 - lightness) * (pixel_values - 1) + 1
        else:
            pos = 2 * (1 + lightness) * pixel_values - 1
        
        # Apply blend3 function across all values
        neg_mask = pos < 0
        pos_mask = pos > 0
        zero_mask = (pos == 0)
        
        # Handle negative positions
        if np.any(neg_mask):
            new_colors[neg_mask] = blend2(black, color, pos[neg_mask] + 1)
        
        # Handle positive positions
        if np.any(pos_mask):
            new_colors[pos_mask] = blend2(color, white, pos[pos_mask])
        
        # Handle zero positions
        if np.any(zero_mask):
            new_colors[zero_mask] = color
    
    # Ensure values are in valid range
    new_colors = np.clip(new_colors, 0, 255).astype(result.dtype)
    
    # Update result image
    result[mask_bool] = new_colors
    
    return result

def create_circular_mask(image, radius_percent=0.25):
    """Create a circular mask in the center of the image"""
    height, width = image.shape[:2]
    center_y, center_x = height // 2, width // 2
    radius = int(min(center_y, center_x) * radius_percent)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    
    return mask

def create_threshold_mask(image, threshold=50, invert=True):
    """
    Create a binary mask using thresholding
    
    Args:
        image: Input image
        threshold: Pixel values below this will be masked (0-255)
        invert: Whether to invert the mask after thresholding
    
    Returns:
        Binary mask where 255 indicates masked regions
    """
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold to create binary mask
    _, binary_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Invert the mask if requested
    if invert:
        binary_mask = cv2.bitwise_not(binary_mask)
    
    return binary_mask

def hsv_to_rgb(h, s, v):
    """Convert HSV values to RGB color array"""
    # Convert h from 0-360 to 0-180 for OpenCV
    h_cv = h / 2
    
    # Create a small image with the HSV color
    hsv = np.uint8([[[h_cv, s * 255, v * 255]]])
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
    
    return rgb

def main():
    parser = argparse.ArgumentParser(description='Apply colorize effect to masked regions of an image')
    parser.add_argument('images', nargs='+', help='Path to the input image(s) or directory')
    parser.add_argument('--output-dir', '-o', help='Directory to save output images', default='./colorized_output')
    parser.add_argument('--mask', '-m', help='Path to the mask image (optional, will create threshold mask if not provided)')
    parser.add_argument('--mask-type', choices=['threshold', 'circle'], default='threshold',
                        help='Type of mask to create if no mask file provided')
    parser.add_argument('--threshold', type=int, default=50, help='Threshold value for mask creation (0-255)')
    parser.add_argument('--invert-mask', action='store_true', help='Invert the mask after creation')
    parser.add_argument('--hue', type=int, help='Hue value (0-360)', default=0)
    parser.add_argument('--saturation', type=float, help='Saturation value (0-1)', default=0.7)
    parser.add_argument('--lightness', type=float, help='Lightness value (-1 to 1)', default=0.0)
    parser.add_argument('--display', action='store_true', help='Display images')
    parser.add_argument('--vectorized', action='store_true', help='Use vectorized implementation')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each input path
    for path in args.images:
        if os.path.isdir(path):
            # Process all images in directory
            for filename in os.listdir(path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    process_image(os.path.join(path, filename), args)
        elif os.path.isfile(path):
            # Process a single image file
            process_image(path, args)
        else:
            print(f"Error: '{path}' is not a valid file or directory")

if __name__ == "__main__":
    main()