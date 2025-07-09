# Video Processing Module

## Overview

This module contains a comprehensive video processing pipeline designed to extract high-quality frames from underwater research videos for prawn counting and measurement analysis. The system intelligently filters frames based on blur detection, similarity checking, and black frame detection to create a clean dataset for machine learning applications.

## Key Features

- **Intelligent Frame Extraction**: Automatically extracts frames from videos while filtering out low-quality content
- **Multi-Quality Checks**: Implements blur detection, similarity checking, and black frame detection
- **Batch Processing**: Supports processing multiple videos in sequence
- **Statistical Logging**: Generates detailed statistics about frame processing decisions
- **Configurable Parameters**: Highly customizable thresholds and processing options
- **EXIF Metadata Preservation**: Maintains timestamp and camera information in extracted frames

## Files Documentation

### 1. `video2dataset.py` (365 lines)

**Primary Purpose**: Main processing engine that converts video files into filtered frame datasets.

**Key Functionality**:
- **Core Class**: `Video2Dataset` - Orchestrates the entire video-to-dataset conversion process
- **Frame Processing Pipeline**: 
  ```python
  # Process each frame through quality checks:
  # 1. Resize for processing efficiency (internal_resolution = 800px)
  # 2. Blur detection using Laplacian variance
  # 3. Black frame detection using luminance analysis
  # 4. Similarity detection using optical flow
  ```
- **Quality Filtering**: Uses three specialized checkers to ensure frame quality:
  - `ThresholdBlurChecker`: Filters blurry frames using Laplacian variance
  - `SimilarityChecker`: Removes duplicate/similar frames using optical flow
  - `BlackFrameChecker`: Excludes dark/black frames using luminance thresholds

**What I did differently**: I implemented a comprehensive processing pipeline that combines multiple quality checks in sequence. The system uses `cv2.Laplacian(image_bw, cv2.CV_64F).var()` for blur detection and `cv2.calcOpticalFlowPyrLK()` for similarity detection, ensuring only high-quality, distinct frames are saved.

**Key Methods**:
- `ProcessVideo()`: Main orchestration method that processes entire videos
- `ProcessFrame()`: Individual frame analysis and quality checking
- `SaveFrame()`: Saves qualified frames with proper EXIF metadata
- `WriteStats()`: Logs processing statistics to CSV file

### 2. `parameters.py` (48 lines)

**Primary Purpose**: Configuration management system for video processing parameters.

**Key Functionality**:
- **Parameter Handling**: Centralizes all processing configuration in a single class
- **Directory Management**: Automatically creates output directories if they don't exist
- **Input Validation**: Handles both single video files and multiple video lists
- **Default Values**: Provides sensible defaults for all optional parameters

**What I did differently**: I created a robust parameter management system that uses `args.get("parameter_name", default_value)` pattern for safe parameter access. The system automatically converts single input strings to lists: `self.input = [self.input] if isinstance(self.input, str) else self.input`.

**Key Parameters**:
```python
# Core processing parameters
self.blur_threshold = 100        # Laplacian variance threshold
self.distance_threshold = 30     # Optical flow distance threshold  
self.internal_resolution = 800   # Processing resolution for efficiency
self.frame_format = "jpg"        # Output image format
```

### 3. `checkers.py` (137 lines)

**Primary Purpose**: Implements specialized quality checking algorithms for frame filtering.

**Key Classes**:

#### `ThresholdBlurChecker`
**What I did differently**: Uses Laplacian variance method for blur detection: `var = cv2.Laplacian(image_bw, cv2.CV_64F).var()`. This approach calculates the variance of the Laplacian filter response, where low variance indicates blur.

```python
def IsBlur(self, image_bw, id, frame):
    var = cv2.Laplacian(image_bw, cv2.CV_64F).var()
    is_blur = var < self.threshold  # Default threshold: 100
    return var, is_blur
```

#### `SimilarityChecker`
**What I did differently**: Implements optical flow-based similarity detection using `cv2.calcOpticalFlowPyrLK()`. The system tracks features between consecutive frames and calculates average movement distance.

```python
def IsSimilar(self, image_bw, id):
    # Track features using optical flow
    features, status, _ = cv2.calcOpticalFlowPyrLK(self.last_image, image_bw, self.last_image_features, None)
    # Calculate average movement distance
    distance = np.average(np.abs(good_features2 - good_features))
```

#### `BlackFrameChecker`
**What I did differently**: Implements a sophisticated two-pass black frame detection. First pass calculates luminance statistics across the entire video, second pass uses these statistics for accurate black pixel detection.

```python
def PreProcess(self, video_path, start_frame, end_frame):
    # Calculate luminance range and minimum values across entire video
    self.absolute_threshold = self.luminance_minimum_value + self.pixel_black_th * self.luminance_range_size
```

### 4. `video_splt_main.py` (35 lines)

**Primary Purpose**: Entry point script for batch video processing operations.

**Key Functionality**:
- **Batch Processing Setup**: Configures processing parameters for multiple videos
- **File Discovery**: Uses `glob.glob()` to find video files matching patterns
- **Processing Orchestration**: Iterates through video files and processes each one

**What I did differently**: I created a flexible batch processing system that uses glob patterns to find videos: `glob.glob(r"C:\Users\...\*.mp4")` and processes them sequentially with identical parameters.

**Configuration Example**:
```python
args = { 
    "blur_threshold": 100,           # Blur detection sensitivity
    "distance_threshold": 30,        # Similarity detection threshold
    "frame_format": "jpg",           # Output format
    "stats_file": "stats.csv"        # Statistics logging
}
```

### 5. `run_video_split.ipynb` (287 lines)

**Primary Purpose**: Interactive Jupyter notebook for video processing experimentation and batch operations.

**Key Sections**:

#### Cell 1: Main Video Processing
**What I did differently**: Implements the same processing logic as the main script but in an interactive environment. Uses the same parameter configuration but allows for real-time parameter adjustment and immediate feedback.

#### Cell 3: Image Quality Detection
**What I did differently**: I added an experimental `ImageQualityDetector` class that uses SSIM and PSNR metrics for quality assessment:
```python
def is_high_quality(self, image):
    ssim_score = ssim(gray, gray)
    psnr_score = psnr(gray, gray)
    return ssim_score > self.ssim_threshold and psnr_score > self.psnr_threshold
```

#### Cell 5: Frame Extraction
**What I did differently**: Implements simple frame extraction every N frames: `if frame_count % 5 == 0:` - This demonstrates alternative extraction strategies beyond quality-based filtering.

#### Cell 8: Image Enhancement
**What I did differently**: Begins implementation of contrast stretching for image enhancement using channel-wise processing with `cv2.split(image)`.

### 6. `extract_frame_gyroflow.ipynb` (Empty file)

**Primary Purpose**: Placeholder for future Gyroflow integration for camera stabilization.

**Status**: Currently empty - likely intended for future development of stabilized frame extraction using Gyroflow data.

## Usage Examples

### Basic Video Processing
```python
from parameters import Parameters
from video2dataset import Video2Dataset

args = {
    "input": "path/to/video.mp4",
    "output": "path/to/output/folder",
    "blur_threshold": 100,
    "distance_threshold": 30,
    "stats_file": "processing_stats.csv"
}

processor = Video2Dataset(Parameters(args))
processor.ProcessVideo()
```

### Batch Processing Multiple Videos
```python
import glob

video_paths = glob.glob("videos/*.mp4")
for video_path in video_paths:
    args["input"] = video_path
    processor = Video2Dataset(Parameters(args))
    processor.ProcessVideo()
```

## Output Structure

The system generates:
- **Filtered Frame Images**: High-quality frames saved as JPG/PNG files
- **Statistics CSV**: Detailed processing log with quality metrics
- **EXIF Metadata**: Preserved timestamp and camera information
- **Processing Log**: Detailed logging of all processing decisions

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `blur_threshold` | 100 | Laplacian variance threshold for blur detection |
| `distance_threshold` | 30 | Optical flow distance for similarity detection |
| `internal_resolution` | 800 | Processing resolution for efficiency |
| `frame_format` | "jpg" | Output image format |
| `black_ratio_threshold` | 0.98 | Ratio of black pixels for black frame detection |
| `pixel_black_threshold` | 0.30 | Luminance threshold for black pixel detection |

## Dependencies

- OpenCV (`cv2`) - Video processing and computer vision
- NumPy (`numpy`) - Numerical computations
- PIL (`Pillow`) - Image handling and EXIF metadata
- scikit-image (`skimage`) - Image quality metrics
- piexif - EXIF metadata manipulation

## Performance Considerations

- **Internal Resolution**: Frames are resized to 800px for processing efficiency
- **Memory Management**: Processes frames sequentially to minimize memory usage
- **Quality vs Speed**: Adjust thresholds based on quality requirements vs processing speed needs
- **Batch Processing**: Process multiple videos in sequence for efficiency 