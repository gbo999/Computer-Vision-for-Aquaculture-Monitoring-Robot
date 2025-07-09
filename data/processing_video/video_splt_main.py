"""
Video Processing Main Script - Batch Video Processing Entry Point

This script serves as the main entry point for batch video processing operations.
It demonstrates how to configure and execute the video-to-dataset conversion
pipeline for multiple videos with consistent parameters.

Key Features:
- Batch processing configuration
- Video file discovery using glob patterns
- Consistent parameter application across videos
- Processing loop with error handling
- Statistical logging and progress tracking

Usage:
    python video_splt_main.py

Configuration:
    Modify the 'args' dictionary to adjust processing parameters
    Update glob patterns to match your video file locations

Author: Research Team
Purpose: Underwater prawn counting and measurement research
"""

import os
from src.data.data_processing.parameters import Parameters
from src.data.data_processing.video2dataset import Video2Dataset
import glob

# Set working directory for relative path operations
work_dir = os.getcwd()

# PROCESSING CONFIGURATION
# This dictionary contains all parameters for video processing
# Modify these values to customize the processing behavior
args = { 
    # INPUT/OUTPUT CONFIGURATION
    "input": f"C:/Users/gbo10/Dropbox/research videos/31.12/65-31.12/GX010065.MP4",  # Default input (will be overridden in loop)
    "output": r"C:\Users\gbo10\Dropbox\research videos\for ice\try",  # Output directory for processed frames
    
    # FRAME RANGE SETTINGS
    "start": 0,           # Starting frame index (0 = beginning of video)
    "end": None,          # Ending frame index (None = process entire video)
    "limit": None,        # Maximum number of output frames (None = unlimited)
    
    # OUTPUT FORMAT SETTINGS
    "output_resolution": None,  # Override output resolution (None = original)
    "frame_format": "jpg",      # Output image format (jpg, png, tiff, etc.)
    
    # QUALITY FILTERING THRESHOLDS
    # These parameters control which frames are accepted/rejected
    
    # Blur Detection: Uses Laplacian variance method
    # Lower values = more sensitive to blur (stricter filtering)
    # Typical range: 50-300, where:
    #   - 50-100: Very strict (only very sharp frames)
    #   - 100-200: Moderate (good balance)
    #   - 200-300: Lenient (accepts slightly blurry frames)
    "blur_threshold": 100,
    
    # Similarity Detection: Uses optical flow distance
    # Lower values = more sensitive to similarity (stricter deduplication)
    # Typical range: 10-50, where:
    #   - 10-20: Very strict (removes similar frames aggressively)
    #   - 20-40: Moderate (good balance of diversity and coverage)
    #   - 40-50: Lenient (only removes very similar frames)
    "distance_threshold": 30,
    
    # Black Frame Detection (currently disabled)
    # These parameters would control black frame detection if enabled
    "black_ratio_threshold": None,   # Ratio of black pixels (0.0-1.0, e.g., 0.98 = 98%)
    "pixel_black_threshold": None,   # Individual pixel black threshold (0.0-1.0)
    
    # METADATA AND LOGGING
    "use-srt": None,  # Enable SRT file processing for GPS/camera metadata (currently disabled)
    "stats_file": "C:/Users/gbo10/Videos/research/counting_research_algorithms/src/data/data_processing/stats.csv"  # Statistics output file
}

# BATCH PROCESSING SETUP
# Configure which videos to process using glob patterns

# Initialize video paths list for batch processing
video_paths = []  # This will store all video files to be processed

# VIDEO FILE DISCOVERY
# Use glob patterns to find video files matching specific criteria
# Modify the glob pattern to match your video file locations and naming conventions

# Example: Process specific video file(s)
# The current pattern looks for a specific file with a complex name pattern
for video_path in glob.glob(r"C:\Users\gbo10\Dropbox\research videos\for ice\GX0102422222222222_Video.mp4"):
    video_paths.append(video_path)
    print(f"Found video for processing: {video_path}")

# Additional glob patterns you might use:
# 
# Process all MP4 files in a directory:
# for video_path in glob.glob(r"C:\path\to\videos\*.mp4"):
#     video_paths.append(video_path)
#
# Process videos with specific naming pattern:
# for video_path in glob.glob(r"C:\path\to\videos\GX01*.MP4"):
#     video_paths.append(video_path)
#
# Process videos from multiple directories:
# for pattern in [r"C:\dir1\*.mp4", r"C:\dir2\*.MP4"]:
#     for video_path in glob.glob(pattern):
#         video_paths.append(video_path)

print(f"Total videos found for processing: {len(video_paths)}")

# BATCH PROCESSING EXECUTION
# Process each discovered video with the same configuration parameters

for i, video_path in enumerate(video_paths, 1):
    # Update input path for current video
    args["input"] = video_path
    
    print(f'\n{"="*60}')
    print(f'Processing video {i}/{len(video_paths)}: {os.path.basename(video_path)}')
    print(f'Full path: {video_path}')
    print(f'{"="*60}')
    
    try:
        # Create parameter object with current configuration
        parameters = Parameters(args)
        
        # Create video processor instance
        video_processor = Video2Dataset(parameters)
        
        # Execute video processing pipeline
        # This will:
        # 1. Load and analyze the video
        # 2. Apply quality filtering (blur, similarity, black frame detection)
        # 3. Save qualified frames with proper naming and metadata
        # 4. Generate processing statistics
        output_paths = video_processor.ProcessVideo()
        
        print(f"Successfully processed {len(output_paths) if output_paths else 0} frames from {os.path.basename(video_path)}")
        
    except Exception as e:
        print(f"ERROR processing video {video_path}: {str(e)}")
        print("Continuing with next video...")
        continue

print(f'\n{"="*60}')
print("BATCH PROCESSING COMPLETED")
print(f"Processed {len(video_paths)} video(s)")
print(f'Output directory: {args["output"]}')
print(f'Statistics file: {args["stats_file"]}')
print(f'{"="*60}')

# POST-PROCESSING NOTES:
# 
# After running this script, you should have:
# 1. Processed frame images in the output directory
# 2. A statistics CSV file with processing details for analysis
# 3. Log files with detailed processing information
#
# Next steps might include:
# - Analyzing the statistics file to optimize parameters
# - Reviewing sample frames to validate quality filtering
# - Running additional processing on the extracted frames
# - Creating datasets for machine learning training