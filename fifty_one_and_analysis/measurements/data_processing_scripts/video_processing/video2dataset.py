"""
Video2Dataset Module - Core Video Processing Engine

This module provides the main functionality for converting video files into filtered
frame datasets for computer vision applications. It implements intelligent quality
filtering using blur detection, similarity checking, and black frame detection.

Key Features:
- Multi-stage quality filtering pipeline
- Configurable processing parameters
- Statistical logging and analysis
- EXIF metadata preservation
- Batch processing capabilities

Author: Research Team
Purpose: Underwater prawn counting and measurement research

This module includes code derived from OpenDroneMap (https://github.com/OpenDroneMap/ODM),
which is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
As such, this module is also licensed under AGPL-3.0.
"""

import datetime
from fractions import Fraction
import io
from math import ceil, floor
import time
import cv2
import os
import collections
from PIL import Image
import numpy as np
import logging
import piexif

from src.data.data_processing.checkers import ThresholdBlurChecker, SimilarityChecker, BlackFrameChecker
from src.data.data_processing.parameters import Parameters

# Configure logging for processing tracking
work_dir=os.getcwd
logging.basicConfig(level=logging.INFO,filename='video2dataset.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Video2Dataset:
    """
    Main class for converting video files into filtered frame datasets.
    
    This class orchestrates the entire video processing pipeline, applying multiple
    quality checks to ensure only high-quality, distinct frames are extracted from
    source videos. The processing pipeline includes:
    
    1. Frame extraction from video files
    2. Blur detection using Laplacian variance
    3. Black frame detection using luminance analysis
    4. Similarity detection using optical flow
    5. Frame saving with EXIF metadata preservation
    6. Statistical logging for analysis
    
    Attributes:
        parameters (Parameters): Configuration object containing all processing settings
        blur_checker (ThresholdBlurChecker): Optional blur detection instance
        similarity_checker (SimilarityChecker): Optional similarity detection instance
        black_checker (BlackFrameChecker): Optional black frame detection instance
        frame_index (int): Current frame index being processed
        global_idx (int): Global frame counter across all processed videos
        date_now (datetime): Pseudo timestamp for frame metadata
        f (file): Statistics file handle for CSV logging
    """

    def __init__(self, parameters : Parameters):
        """
        Initialize the Video2Dataset processor with configuration parameters.
        
        Args:
            parameters (Parameters): Configuration object containing processing settings
                including thresholds, input/output paths, and quality check options
        """
        self.parameters = parameters

        # Initialize quality checkers based on provided thresholds
        # Each checker is optional and only created if threshold is specified
        self.blur_checker = ThresholdBlurChecker(parameters.blur_threshold) if parameters.blur_threshold is not None else None
        self.similarity_checker = SimilarityChecker(parameters.distance_threshold) if parameters.distance_threshold is not None else None
        # self.black_checker = BlackFrameChecker(parameters.black_ratio_threshold, parameters.pixel_black_threshold) if parameters.black_ratio_threshold is not None or parameters.pixel_black_threshold is not None else None
        self.black_checker = None  # Currently disabled - can be enabled for black frame detection
        
        # Initialize frame tracking
        self.frame_index = parameters.start
        
        print(self.parameters.stats_file)

    def ProcessVideo(self):
        """
        Main processing method that handles the complete video-to-dataset conversion.
        
        This method orchestrates the entire processing pipeline:
        1. Opens and validates input video files
        2. Initializes quality checkers and preprocessing
        3. Processes each frame through the quality pipeline
        4. Saves qualified frames with proper metadata
        5. Generates processing statistics
        
        Returns:
            list: List of paths to successfully processed frame files
            
        Processing Pipeline:
        - Frame Extraction: Read frames sequentially from video
        - Quality Checks: Apply blur, similarity, and black frame detection
        - Frame Saving: Save qualified frames with EXIF metadata
        - Statistics: Log all processing decisions for analysis
        """
        self.date_now = None
        start = time.time()
        logging.info("Processing video")
        logging.info('stats file is {}'.format(self.parameters.stats_file))
        
        # Initialize statistics file for processing logging
        if (self.parameters.stats_file is not None):
            logger.info("Writing stats to file: {}".format(self.parameters.stats_file))
            self.f = open(self.parameters.stats_file, "w") 
            # CSV header with all tracked metrics
            self.f.write("global_idx;file_name;frame_index;blur_score;is_blurry;is_black;last_frame_index;similarity_score;is_similar;written\n")
        else:
            return

        self.global_idx = 0  # Global frame counter across all videos
        output_file_paths = []  # Track all successfully processed frames
        
        # Process each input video file
        for input_file in self.parameters.input:
            file_name = os.path.basename(input_file)
            logger.info("Processing video: {}".format(input_file))

            # Extract video metadata for processing
            video_info = get_video_info(input_file)
            logger.info(video_info)

            # Set pseudo start time for EXIF metadata
            # Use file modification time or current time as baseline
            if self.date_now is None:
                try:
                    self.date_now = datetime.datetime.fromtimestamp(os.path.getmtime(input_file))
                except:
                    self.date_now = datetime.datetime.now()
            else:
                # For multiple videos, advance timestamp based on video duration
                self.date_now += datetime.timedelta(seconds=video_info.total_frames / video_info.frame_rate)
            
            logger.info("Use pseudo start time: %s" % self.date_now)

            # SRT file processing is currently disabled but framework exists
            # TODO: Re-enable SRT processing for GPS and camera metadata extraction
            srt_parser = None

            # Preprocessing for black frame checker (computationally intensive)
            # Analyzes entire video to calculate luminance statistics
            if (self.black_checker is not None and self.black_checker.NeedPreProcess()):
                start2 = time.time()
                logger.info("Preprocessing for black frame checker... this might take a bit")
                self.black_checker.PreProcess(input_file, self.parameters.start, self.parameters.end)
                end = time.time()
                logger.info("Preprocessing time: {:.2f}s".format(end - start2))
                logger.info("Calculated luminance_range_size is {}".format(self.black_checker.luminance_range_size))
                logger.info("Calculated luminance_minimum_value is {}".format(self.black_checker.luminance_minimum_value))
                logger.info("Calculated absolute_threshold is {}".format(self.black_checker.absolute_threshold))

            # Open video file for frame-by-frame processing
            cap = cv2.VideoCapture(input_file)
            if (not cap.isOpened()):
                logger.info("Error opening video stream or file")
                return

            # Set starting frame position if specified
            if (self.parameters.start is not None):
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.parameters.start)
                self.frame_index = self.parameters.start
                start_frame = self.parameters.start
            else:
                start_frame = 0

            # Calculate total frames to process for progress tracking
            frames_to_process = self.parameters.end - start_frame + 1 if (self.parameters.end is not None) else video_info.total_frames - start_frame

            progress = 0
            # Main frame processing loop
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:  # End of video
                    break

                if (self.parameters.end is not None and self.frame_index > self.parameters.end):
                    break  # Reached specified end frame

                # Update and display progress
                prev_progress = progress
                progress = floor((self.frame_index - start_frame + 1) / frames_to_process * 100)
                if progress != prev_progress:
                    print("[{}][{:3d}%] Processing frame {}/{}: ".format(file_name, progress, self.frame_index - start_frame + 1, frames_to_process), end="\r")

                # Process individual frame through quality pipeline
                stats = self.ProcessFrame(frame, video_info, srt_parser)

                # Log processing statistics
                if stats is not None and self.parameters.stats_file is not None:
                    self.WriteStats(input_file, stats)

                # Track successfully processed frames
                if stats is not None and "written" in stats.keys():
                    output_file_paths.append(stats["path"])

            cap.release()

        # Close statistics file
        if self.f is not None:
            self.f.close()

        # Apply limit if specified (randomly sample from processed frames)
        if self.parameters.limit is not None and self.parameters.limit > 0 and self.global_idx >= self.parameters.limit:
            logger.info("Limit of {} frames reached, trimming dataset".format(self.parameters.limit))
            output_file_paths = limit_files(output_file_paths, self.parameters.limit)

        end = time.time()
        logger.info("Total processing time: {:.2f}s".format(end - start))
        return output_file_paths

    def ProcessFrame(self, frame, video_info, srt_parser):
        """
        Process a single frame through the complete quality checking pipeline.
        
        This method applies all configured quality checks in sequence:
        1. Convert to grayscale and resize for processing efficiency
        2. Blur detection using Laplacian variance
        3. Black frame detection using luminance analysis  
        4. Similarity detection using optical flow
        5. Frame saving if all checks pass
        
        Args:
            frame (numpy.ndarray): Raw frame from video (BGR format)
            video_info (VideoInfo): Video metadata including frame rate and dimensions
            srt_parser: SRT file parser for metadata extraction (currently unused)
            
        Returns:
            dict: Processing statistics including quality scores and decisions
                Keys: frame_index, global_idx, blur_score, is_blurry, is_black,
                     similarity_score, is_similar, last_frame_index, written, path
        """
        # Initialize result dictionary with basic frame info
        res = {"frame_index": self.frame_index, "global_idx": self.global_idx}

        # Convert to grayscale for quality analysis
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frame for processing efficiency
        # This significantly speeds up quality checks while maintaining accuracy
        h, w = frame_bw.shape
        resolution = self.parameters.internal_resolution  # Default: 800 pixels
        if resolution < w or resolution < h:
            m = max(w, h)
            factor = resolution / m
            frame_bw = cv2.resize(frame_bw, (int(ceil(w * factor)), int(ceil(h * factor))), interpolation=cv2.INTER_NEAREST)

        # QUALITY CHECK 1: Blur Detection
        # Uses Laplacian variance - low variance indicates blur
        if (self.blur_checker is not None):
            blur_score, is_blurry = self.blur_checker.IsBlur(frame_bw, self.frame_index, frame)
            res["blur_score"] = blur_score
            res["is_blurry"] = is_blurry

            if is_blurry:
                # Skip blurry frames - they're not useful for analysis
                self.frame_index += 1
                return res

        # QUALITY CHECK 2: Black Frame Detection  
        # Identifies frames that are predominantly black/dark
        if (self.black_checker is not None):
            is_black = self.black_checker.IsBlack(frame_bw, self.frame_index)
            res["is_black"] = is_black

            if is_black:
                # Skip black frames - they contain no useful information
                self.frame_index += 1
                return res

        # QUALITY CHECK 3: Similarity Detection
        # Uses optical flow to detect frames too similar to previous ones
        if (self.similarity_checker is not None):
            similarity_score, is_similar, last_frame_index = self.similarity_checker.IsSimilar(frame_bw, self.frame_index)
            res["similarity_score"] = similarity_score
            res["is_similar"] = is_similar
            res["last_frame_index"] = last_frame_index

            if is_similar:
                # Skip similar frames to avoid dataset redundancy
                self.frame_index += 1
                return res

        # FRAME ACCEPTED: Save frame with metadata
        path = self.SaveFrame(frame, video_info, srt_parser)
        res["written"] = True
        res["path"] = path
        self.frame_index += 1
        self.global_idx += 1

        return res

    def SaveFrame(self, frame, video_info, srt_parser):
        """
        Save a qualified frame with proper naming, resizing, and EXIF metadata.
        
        This method handles the final frame processing and saving:
        1. Optional resizing based on max_dimension parameter
        2. Generate systematic filename with video info and frame indices
        3. Create and embed EXIF metadata with timestamps
        4. Save as high-quality image file
        
        Args:
            frame (numpy.ndarray): Frame to save (BGR format)
            video_info (VideoInfo): Video metadata for filename generation
            srt_parser: SRT parser for GPS metadata (currently unused)
            
        Returns:
            str: Full path to saved frame file
            
        Filename Format: {video_basename}_{global_idx}_{frame_index}.{format}
        Example: GX010191_042_1250.jpg
        """
        # Apply output resolution limit if specified
        max_dim = self.parameters.max_dimension
        if max_dim is not None:
            h, w, _ = frame.shape
            if max_dim < w or max_dim < h:
                m = max(w, h)
                factor = max_dim / m
                frame = cv2.resize(frame, (int(ceil(w * factor)), int(ceil(h * factor))), interpolation=cv2.INTER_AREA)

        # Generate systematic filename with video and frame information
        path = os.path.join(self.parameters.output,
            "{}_{}_{}.{}".format(video_info.basename, self.global_idx, self.frame_index, self.parameters.frame_format))

        # Encode frame for metadata processing
        _, buf = cv2.imencode('.' + self.parameters.frame_format, frame)

        # Calculate elapsed time for timestamp metadata
        delta = datetime.timedelta(seconds=(self.frame_index / video_info.frame_rate))
        elapsed_time = datetime.datetime(1900, 1, 1) + delta

        img = Image.open(io.BytesIO(buf))
        
        # SRT GPS data processing (currently disabled)
        # entry = gps_coords = None
        # if srt_parser is not None:
        #     entry = srt_parser.get_entry(elapsed_time)
        #     gps_coords = srt_parser.get_gps(elapsed_time)

        # Calculate absolute timestamp for EXIF metadata
        exif_time = (elapsed_time + (self.date_now - datetime.datetime(1900, 1, 1)))
        elapsed_time_str = exif_time.strftime("%Y:%m:%d %H:%M:%S")
        subsec_time_str = exif_time.strftime("%f")

        # Create comprehensive EXIF metadata dictionary
        # Preserves timing and resolution information for research traceability
        exif_dict = {
            "0th": {
                # Basic image metadata
                piexif.ImageIFD.DateTime: elapsed_time_str,
                piexif.ImageIFD.XResolution: (frame.shape[1], 1),
                piexif.ImageIFD.YResolution: (frame.shape[0], 1),
            },
            "Exif": {
                # Detailed timestamp and dimension metadata
                piexif.ExifIFD.DateTimeOriginal: elapsed_time_str,
                piexif.ExifIFD.DateTimeDigitized: elapsed_time_str,
                piexif.ExifIFD.SubSecTime: subsec_time_str,
                piexif.ExifIFD.PixelXDimension: frame.shape[1],
                piexif.ExifIFD.PixelYDimension: frame.shape[0],
            }}

        # Additional camera metadata processing (framework exists)
        # Can be extended for camera settings, GPS coordinates, etc.
        
        # Save image with embedded EXIF metadata
        exif_bytes = piexif.dump(exif_dict)
        img.save(path, exif=exif_bytes, quality=95)  # High quality for research

        return path

    def WriteStats(self, input_file, stats):
        """
        Write processing statistics to CSV file for analysis and debugging.
        
        This method logs all processing decisions and quality metrics for each frame,
        enabling post-processing analysis and parameter optimization.
        
        Args:
            input_file (str): Path to source video file
            stats (dict): Processing statistics dictionary containing all quality metrics
            
        CSV Format: global_idx;file_name;frame_index;blur_score;is_blurry;is_black;
                   last_frame_index;similarity_score;is_similar;written
        """
        self.f.write("{};{};{};{};{};{};{};{};{};{}\n".format(
            stats["global_idx"],
            input_file,
            stats["frame_index"],
            stats["blur_score"] if "blur_score" in stats else "",
            stats["is_blurry"] if "is_blurry" in stats else "",
            stats["is_black"] if "is_black" in stats else "",
            stats["last_frame_index"] if "last_frame_index" in stats else "",
            stats["similarity_score"] if "similarity_score" in stats else "",
            stats["is_similar"] if "is_similar" in stats else "",
            stats["written"] if "written" in stats else "").replace(".", ","))  # European CSV format


def get_video_info(input_file):
    """
    Extract essential metadata from video file for processing.
    
    Args:
        input_file (str): Path to video file
        
    Returns:
        VideoInfo: Named tuple containing total_frames, frame_rate, and basename
    """
    video = cv2.VideoCapture(input_file)
    basename = os.path.splitext(os.path.basename(input_file))[0]

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video.get(cv2.CAP_PROP_FPS)

    video.release()

    return collections.namedtuple("VideoInfo", ["total_frames", "frame_rate", "basename"])(total_frames, frame_rate, basename)

def float_to_rational(f):
    """Convert float to rational number for EXIF metadata."""
    f = Fraction(f).limit_denominator()
    return (f.numerator, f.denominator)

def limit_files(paths, limit):
    """
    Limit dataset size by selecting evenly distributed frames and removing others.
    
    This function implements intelligent dataset limiting by:
    1. Selecting evenly distributed frames across the entire dataset
    2. Physically deleting non-selected frames to save storage
    3. Returning paths to kept frames
    
    Args:
        paths (list): List of all frame file paths
        limit (int): Maximum number of frames to keep
        
    Returns:
        list: Paths to frames that were kept
    """
    if len(paths) <= limit:
        return paths
    
    to_keep = []
    all_idxes = np.arange(0, len(paths))
    keep_idxes = np.linspace(0, len(paths) - 1, limit, dtype=int)  # Even distribution
    remove_idxes = set(all_idxes) - set(keep_idxes)

    p = np.array(paths)
    to_keep = list(p[keep_idxes])

    # Remove excess files to save storage space
    for idx in remove_idxes:
        os.remove(paths[idx])

    return to_keep

# GPS and camera metadata processing framework (currently disabled)
# These functions provide the structure for future GPS and camera data integration

# def to_deg(value, loc):
#     """convert decimal coordinates into degrees, munutes and seconds tuple
#     Keyword arguments: value is float gps-value, loc is direction list ["S", "N"] or ["W", "E"]
#     return: tuple like (25, 13, 48.343 ,'N')
#     """
#     if value < 0:
#         loc_value = loc[0]
#     elif value > 0:
#         loc_value = loc[1]
#     else:
#         loc_value = ""
#     abs_value = abs(value)
#     deg =  int(abs_value)
#     t1 = (abs_value-deg)*60
#     min = int(t1)
#     sec = round((t1 - min)* 60, 5)
#     return (deg, min, sec, loc_value)

# def get_gps_location(elapsed_time, lat, lng, altitude):
#     """Generate GPS EXIF metadata from coordinates."""
#     lat_deg = to_deg(lat, ["S", "N"])
#     lng_deg = to_deg(lng, ["W", "E"])

#     exiv_lat = (float_to_rational(lat_deg[0]), float_to_rational(lat_deg[1]), float_to_rational(lat_deg[2]))
#     exiv_lng = (float_to_rational(lng_deg[0]), float_to_rational(lng_deg[1]), float_to_rational(lng_deg[2]))

#     gps_ifd = {
#         piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
#         piexif.GPSIFD.GPSDateStamp: elapsed_time.strftime('%Y:%m:%d')
#     }

#     if lat is not None and lng is not None:
#         gps_ifd[piexif.GPSIFD.GPSLatitudeRef] = lat_deg[3]
#         gps_ifd[piexif.GPSIFD.GPSLatitude] = exiv_lat
#         gps_ifd[piexif.GPSIFD.GPSLongitudeRef] = lng_deg[3]
#         gps_ifd[piexif.GPSIFD.GPSLongitude] = exiv_lng
#         if altitude is not None:
#             gps_ifd[piexif.GPSIFD.GPSAltitudeRef] = 0
#             gps_ifd[piexif.GPSIFD.GPSAltitude] = float_to_rational(round(altitude))

#     return gps_ifd