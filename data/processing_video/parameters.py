"""
Parameters Module - Configuration Management for Video Processing

This module provides centralized parameter management for the video processing pipeline.
It handles configuration validation, default value assignment, and directory management
for all video-to-dataset conversion operations.

Key Features:
- Centralized configuration management
- Automatic directory creation
- Input validation and normalization
- Default value assignment
- Type conversion and safety checks

Author: Research Team
Purpose: Underwater prawn counting and measurement research
"""

import argparse
import datetime
import os

class Parameters:
    """
    Configuration management class for video processing operations.
    
    This class centralizes all processing parameters, provides default values,
    and handles input validation and normalization. It ensures consistent
    configuration across all video processing operations.
    
    The class handles both single video processing and batch operations,
    automatically normalizing inputs and creating necessary directories.
    
    Supported Parameters:
    - Input/Output paths and formats
    - Quality filtering thresholds  
    - Processing limits and constraints
    - Statistical logging configuration
    - Resolution and format settings
    
    Attributes:
        input (list): List of input video file paths
        output (str): Output directory path for processed frames
        start (int): Starting frame index for processing
        end (int): Ending frame index (None for entire video)
        limit (int): Maximum number of output frames
        blur_threshold (float): Laplacian variance threshold for blur detection
        distance_threshold (float): Optical flow distance threshold for similarity
        black_ratio_threshold (float): Black pixel ratio threshold
        pixel_black_threshold (float): Individual pixel black threshold
        use_srt (bool): Enable SRT file processing for metadata
        frame_format (str): Output image format (jpg, png, etc.)
        max_dimension (int): Maximum output dimension for resizing
        stats_file (str): Path to statistics CSV file
        internal_resolution (int): Processing resolution for efficiency
    """

    def __init__(self, args):
        """
        Initialize Parameters object with configuration dictionary.
        
        This constructor processes the input arguments, applies defaults,
        validates settings, and prepares the configuration for video processing.
        It automatically creates output directories and normalizes input paths.
        
        Args:
            args (dict): Configuration dictionary containing processing parameters
            
        Parameter Descriptions:
            "input": Path to input video file(s), use list for multiple files
            "output": Path to output directory for processed frames
            "start": Start frame index (0-based, default: 0)
            "end": End frame index (None for entire video)
            "output_resolution": Override output resolution (deprecated)
            "blur_threshold": Blur detection threshold - Laplacian variance below this is blurry (good value: 100-300)
            "distance_threshold": Similarity threshold - optical flow distance below this is similar (good value: 20-50)
            "black_ratio_threshold": Black frame detection - ratio of black pixels (default: 0.98 = 98%)
            "pixel_black_threshold": Individual pixel black threshold - luminance percentage (default: 0.30 = 30%)
            "use_srt": Enable SRT file processing for GPS/camera metadata
            "limit": Maximum number of output frames (None for unlimited)
            "frame_format": Output image format - jpg, png, tiff, etc. (default: jpg)
            "stats_file": Path to CSV file for processing statistics logging
            "max_dimension": Maximum output image dimension for resizing (None for original)
        """
        
        # DIRECTORY MANAGEMENT
        # Ensure output directory exists before processing begins
        if not os.path.exists(args["output"]):
            os.makedirs(args["output"])
            print(f"Created output directory: {args['output']}")

        # INPUT PATH NORMALIZATION
        # Convert single video path to list for consistent processing
        # This allows the same processing logic for single and batch operations
        self.input = args["input"]
        if isinstance(self.input, str):
            self.input = [self.input]  # Normalize to list format
            print(f"Normalized single input to list: {self.input}")

        # CORE PROCESSING PARAMETERS
        self.output = args["output"]  # Output directory for processed frames
        
        # FRAME RANGE CONFIGURATION
        # Define which portion of video(s) to process
        self.start = args.get("start", 0)  # Starting frame (0-based indexing)
        self.end = args.get("end", None)   # Ending frame (None = process entire video)
        self.limit = args.get("limit", None)  # Maximum output frames (None = unlimited)
        
        # QUALITY FILTERING THRESHOLDS
        # These parameters control the quality filtering pipeline
        
        # Blur Detection: Uses Laplacian variance method
        # Lower values = more sensitive to blur (stricter filtering)
        # Typical values: 100-300 (100=strict, 300=lenient)
        self.blur_threshold = args.get("blur_threshold", None)
        
        # Similarity Detection: Uses optical flow distance
        # Lower values = more sensitive to similarity (stricter deduplication)
        # Typical values: 20-50 (20=strict, 50=lenient)
        self.distance_threshold = args.get("distance_threshold", None)
        
        # Black Frame Detection: Uses luminance analysis
        # Higher ratio = more pixels must be black to trigger detection
        # Default 0.98 = 98% of pixels must be black
        self.black_ratio_threshold = args.get("black_ratio_threshold", None)
        
        # Individual pixel black threshold (0.0-1.0)
        # Luminance percentage below which a pixel is considered black
        # Default 0.30 = pixels with <30% luminance are black
        self.pixel_black_threshold = args.get("pixel_black_threshold", None)
        
        # METADATA PROCESSING
        # Enable SRT file processing for GPS coordinates and camera settings
        self.use_srt = "use_srt" in args
        
        # OUTPUT FORMAT CONFIGURATION
        self.frame_format = args.get("frame_format", "jpg")  # Image format for saved frames
        self.max_dimension = args.get("max_dimension", None)  # Output resolution limit
        
        # STATISTICS AND LOGGING
        print('before stats file')   
        self.stats_file = args.get("stats_file", None)  # CSV file for processing statistics
        print('after stats file')
        print(self.stats_file) 
        
        # PROCESSING OPTIMIZATION
        # Internal resolution for quality checks - frames are resized to this size
        # during processing for efficiency. This doesn't affect output resolution.
        # Lower values = faster processing but potentially less accurate quality detection
        # Higher values = slower processing but more accurate quality detection
        # Default 800 pixels provides good balance of speed and accuracy
        self.internal_resolution = 800
        
        # PARAMETER VALIDATION AND LOGGING
        self._validate_parameters()
        self._log_configuration()
    
    def _validate_parameters(self):
        """
        Validate parameter values and ensure configuration consistency.
        
        This method performs sanity checks on the provided parameters
        to catch common configuration errors before processing begins.
        """
        # Validate frame range
        if self.start < 0:
            raise ValueError("Start frame must be non-negative")
        
        if self.end is not None and self.end < self.start:
            raise ValueError("End frame must be greater than start frame")
        
        # Validate thresholds
        if self.blur_threshold is not None and self.blur_threshold < 0:
            raise ValueError("Blur threshold must be non-negative")
        
        if self.distance_threshold is not None and self.distance_threshold < 0:
            raise ValueError("Distance threshold must be non-negative")
        
        # Validate output format
        supported_formats = ['jpg', 'jpeg', 'png', 'tiff', 'bmp']
        if self.frame_format.lower() not in supported_formats:
            print(f"Warning: Format '{self.frame_format}' may not be supported. Supported formats: {supported_formats}")
        
        # Validate file paths
        for input_path in self.input:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input video file not found: {input_path}")
    
    def _log_configuration(self):
        """
        Log the current configuration for debugging and documentation.
        
        This method provides a comprehensive overview of all processing
        parameters, making it easier to reproduce results and debug issues.
        """
        print("\n" + "="*50)
        print("VIDEO PROCESSING CONFIGURATION")
        print("="*50)
        print(f"Input files: {len(self.input)} video(s)")
        for i, path in enumerate(self.input):
            print(f"  {i+1}. {path}")
        print(f"Output directory: {self.output}")
        print(f"Frame range: {self.start} to {self.end if self.end else 'end'}")
        print(f"Frame limit: {self.limit if self.limit else 'unlimited'}")
        print(f"Output format: {self.frame_format}")
        if self.max_dimension:
            print(f"Output resolution limit: {self.max_dimension}px")
        
        print("\nQUALITY FILTERING:")
        print(f"  Blur threshold: {self.blur_threshold if self.blur_threshold else 'disabled'}")
        print(f"  Similarity threshold: {self.distance_threshold if self.distance_threshold else 'disabled'}")
        print(f"  Black frame detection: {self.black_ratio_threshold if self.black_ratio_threshold else 'disabled'}")
        
        print("\nPROCESSING SETTINGS:")
        print(f"  Internal resolution: {self.internal_resolution}px")
        print(f"  Statistics file: {self.stats_file if self.stats_file else 'disabled'}")
        print(f"  SRT processing: {'enabled' if self.use_srt else 'disabled'}")
        print("="*50 + "\n")
