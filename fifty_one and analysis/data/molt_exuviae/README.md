# Molt/Exuviae Dataset

## Dataset Overview

This dataset contains images of prawn molts (exuviae) used for growth analysis and measurement validation.

### Directory Structure

```
molt_exuviae/
├── original/           # Original undistorted images
├── colorized/         # Processed images with enhanced visibility
└── binary_masks/      # Binary segmentation masks
```

## Image Types

### 1. Original Images
- **Format**: JPG
- **Resolution**: 5312x2988 pixels
- **Camera**: GoPro Hero 10
- **Naming**: `undistorted_GX010191_[frame]_[timestamp].jpg`
- **Location**: `/original/`

### 2. Colorized Images
- **Format**: PNG
- **Resolution**: Same as original
- **Processing**: Enhanced contrast and color segmentation
- **Naming**: `colorized_GX010191_[frame]_[timestamp].png`
- **Location**: `/colorized/`

### 3. Binary Masks
- **Format**: PNG
- **Resolution**: Same as original
- **Content**: Black and white segmentation masks
- **Naming**: `mask_GX010191_[frame]_[timestamp].png`
- **Location**: `/binary_masks/`

## Processing Pipeline

1. **Image Undistortion**:
   - Using Gyroflow calibration
   - Same camera matrix as measurement dataset

2. **Colorization Process**:
   ```python
   # Pseudo-code for colorization
   def colorize_molt(image):
       # Enhance contrast
       enhanced = cv2.equalizeHist(image)
       # Apply color mapping
       colorized = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
       return colorized
   ```

3. **Binary Mask Generation**:
   - Otsu thresholding
   - Morphological operations
   - Manual corrections where needed

## Usage Examples

1. **Loading Images**:
   ```python
   import cv2
   
   # Load original
   img = cv2.imread('original/undistorted_GX010191_10_370.jpg')
   
   # Load colorized
   colorized = cv2.imread('colorized/colorized_GX010191_10_370.png')
   
   # Load mask
   mask = cv2.imread('binary_masks/mask_GX010191_10_370.png', 0)
   ```

2. **Batch Processing**:
   ```python
   from pathlib import Path
   
   def process_batch(input_dir):
       for img_path in Path(input_dir).glob('*.jpg'):
           # Process each image
           process_single_image(img_path)
   ```

## Related Scripts

1. `binary_molt_colorizer.py`: Main colorization script
2. `analyze_good_detections.py`: Detection analysis
3. `process_molt_images.py`: Batch processing utilities

## Notes

- Images are paired (original-colorized-mask)
- Some molts may be partially damaged
- Lighting conditions are controlled
- Scale markers are included in each image
- Manual verification was performed on all masks 