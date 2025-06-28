#!/usr/bin/env python3
"""Data processing and file I/O utilities for prawn measurement analysis."""

import os
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

from .models import PoseDetection, PrawnMeasurements, BoundingBoxMeasurement
from .config import Config


class DataLoader:
    """
    Handles loading and processing of measurement data files.
    
    This class unifies carapace and body data loading that was previously
    duplicated across multiple functions in data_loader.py.
    """
    
    def __init__(self, measurement_type: str):
        """
        Initialize data loader for specific measurement type.
        
        Args:
            measurement_type: Either 'carapace' or 'body'
        """
        self.measurement_type = measurement_type
        self.config = Config()
    
    def load_carapace_data(self) -> pd.DataFrame:
        """Load carapace measurement data from CSV."""
        return pd.read_csv(self.config.CARAPACE_DATA_PATH)
    
    def load_body_data(self) -> pd.DataFrame:
        """Load body measurement data from Excel."""
        return pd.read_excel(self.config.BODY_DATA_PATH)
    
    def load_metadata(self) -> pd.DataFrame:
        """Load image metadata from Excel."""
        return pd.read_excel(self.config.METADATA_PATH)
    
    def load_test_images_metadata(self) -> pd.DataFrame:
        """Load test images metadata from Excel."""
        return pd.read_excel(self.config.METADATA_PATH)
    
    def get_data(self) -> pd.DataFrame:
        """Get data for the configured measurement type."""
        if self.measurement_type == 'carapace':
            return self.load_carapace_data()
        elif self.measurement_type == 'body':
            return self.load_body_data()
        else:
            raise ValueError(f"Unknown measurement type: {self.measurement_type}")


class YOLOParser:
    """
    Parses YOLO format pose estimation files.
    
    Extracted from multiple parsing functions in data_loader.py to provide
    a unified interface for reading prediction and ground truth files.
    """
    
    @staticmethod
    def parse_file(file_path: Path) -> List[PoseDetection]:
        """
        Parse a YOLO format file into list of detections.
        
        Args:
            file_path: Path to YOLO format file
            
        Returns:
            List of PoseDetection objects
        """
        detections = []
        
        if not file_path.exists():
            return detections
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    detection = PoseDetection.from_yolo_line(line)
                    if detection:
                        detections.append(detection)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return detections
    
    @staticmethod
    def find_best_detection(detections: List[PoseDetection], 
                          target_keypoints: List[int]) -> Optional[PoseDetection]:
        """
        Find the detection with the highest confidence for target keypoints.
        
        Args:
            detections: List of pose detections
            target_keypoints: Indices of keypoints to consider
            
        Returns:
            Best detection or None if no valid detection found
        """
        best_detection = None
        best_confidence = 0
        
        for detection in detections:
            # Check if required keypoints are present with sufficient confidence
            keypoint_confidences = []
            for kp_idx in target_keypoints:
                if kp_idx < len(detection.keypoints):
                    keypoint_confidences.append(detection.keypoints[kp_idx][2])
            
            if keypoint_confidences and all(conf > 0.5 for conf in keypoint_confidences):
                avg_confidence = sum(keypoint_confidences) / len(keypoint_confidences)
                if avg_confidence > best_confidence:
                    best_confidence = avg_confidence
                    best_detection = detection
        
        return best_detection


class FilenameProcessor:
    """
    Processes filenames to extract identifiers and clean names.
    
    Consolidates filename processing logic that was scattered throughout
    the original data_loader.py file.
    """
    
    @staticmethod
    def extract_base_filename(filename: str) -> str:
        """
        Extract base filename by removing extensions and hash suffixes.
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned base filename
        """
        # Remove file extension
        base = Path(filename).stem
        
        # Remove Roboflow hash pattern (e.g., ".rf.d49b41f3c5a08c7aa8fd8a1779b49804")
        base = re.sub(r'\.rf\.[a-f0-9]{32}', '', base)
        
        # Remove other common suffixes
        base = re.sub(r'-jpg_gamma_jpg$', '', base)
        
        return base
    
    @staticmethod
    def extract_prawn_id(filename: str) -> Optional[str]:
        """
        Extract prawn ID from filename using regex patterns.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Extracted prawn ID or None if not found
        """
        # Common patterns for prawn ID extraction
        patterns = [
            r'GX(\d+)_(\d+)_(\d+)',  # GX010191_10_370 format
            r'(\w+)_(\d+)_(\d+)',    # Generic pattern
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return '_'.join(match.groups())
        
        return None
    
    @staticmethod
    def determine_pond_type(filepath: str) -> str:
        """
        Determine pond type from file path.
        
        Args:
            filepath: Full file path
            
        Returns:
            Pond type ('right', 'left', 'car', or 'unknown')
        """
        filepath_lower = filepath.lower()
        
        if 'right' in filepath_lower:
            return 'right'
        elif 'left' in filepath_lower:
            return 'left'
        elif 'car' in filepath_lower:
            return 'car'
        else:
            return 'unknown'


class BoundingBoxProcessor:
    """
    Processes bounding box data and calculations.
    
    Extracted from bounding box processing functions in data_loader.py
    to provide centralized bbox handling.
    """
    
    @staticmethod
    def normalize_bbox(bbox: Tuple[float, float, float, float], 
                      image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """
        Convert normalized bbox coordinates to pixel coordinates.
        
        Args:
            bbox: Normalized bbox (x, y, w, h) in range [0, 1]
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Bbox in pixel coordinates
        """
        x_norm, y_norm, w_norm, h_norm = bbox
        
        x_px = x_norm * image_width
        y_px = y_norm * image_height
        w_px = w_norm * image_width
        h_px = h_norm * image_height
        
        return (x_px, y_px, w_px, h_px)
    
    @staticmethod
    def calculate_bbox_measurements(bbox: Tuple[float, float, float, float], 
                                  length_mm: float) -> BoundingBoxMeasurement:
        """
        Calculate comprehensive measurements for a bounding box.
        
        Args:
            bbox: Bounding box coordinates (x, y, w, h)
            length_mm: Associated length measurement in mm
            
        Returns:
            BoundingBoxMeasurement with calculated metrics
        """
        return BoundingBoxMeasurement.from_bbox_and_length(bbox, length_mm)


class PrawnDataProcessor:
    """
    Processes prawn measurement data from CSV/Excel files.
    
    Consolidates data extraction logic that was repeated across
    carapace and body processing functions.
    """
    
    def __init__(self, measurement_type: str):
        """Initialize processor for specific measurement type."""
        self.measurement_type = measurement_type
        self.data_loader = DataLoader(measurement_type)
    
    def extract_measurements_from_row(self, row: pd.Series) -> Optional[PrawnMeasurements]:
        """
        Extract PrawnMeasurements from a data row.
        
        Args:
            row: Pandas Series containing measurement data
            
        Returns:
            PrawnMeasurements object or None if data is insufficient
        """
        try:
            # Extract basic information
            if self.measurement_type == 'carapace':
                filename = str(row.get('Label', ''))  # Use 'Label' for carapace
                # Strip 'carapace:' prefix and '.jpg_gamma' suffix if present
                if filename.startswith('carapace:'):
                    filename = filename[len('carapace:'):]
                if filename.endswith('.jpg_gamma'):
                    filename = filename[:-len('.jpg_gamma')]
            else:
                filename = str(row.get('Label', ''))  # Use 'Label' for body
            prawn_id = FilenameProcessor.extract_prawn_id(filename)
            pond_type = FilenameProcessor.determine_pond_type(filename)
            
            if not prawn_id:
                return None
            
            # Extract manual measurements based on type
            if self.measurement_type == 'carapace':
                manual_lengths = [
                    float(row.get('Length_1', 0) or 0),
                    float(row.get('Length_2', 0) or 0),
                    float(row.get('Length_3', 0) or 0)
                ]
                manual_scales = [
                    float(row.get('Scale_1', 0) or 0),
                    float(row.get('Scale_2', 0) or 0),
                    float(row.get('Scale_3', 0) or 0)
                ]
            else:  # body measurements
                manual_lengths = [
                    float(row.get('body_length_1', 0) or 0),
                    float(row.get('body_length_2', 0) or 0),
                    float(row.get('body_length_3', 0) or 0)
                ]
                manual_scales = [
                    float(row.get('body_scale_1', 0) or 0),
                    float(row.get('body_scale_2', 0) or 0),
                    float(row.get('body_scale_3', 0) or 0)
                ]
            
            # Filter out zero/invalid measurements
            manual_lengths = [l for l in manual_lengths if l and l > 0]
            manual_scales = [s for s in manual_scales if s and s > 0]
            
            if not manual_lengths:
                return None
            
            # Create bounding box measurements (placeholder for now)
            bounding_boxes = []
            
            return PrawnMeasurements(
                prawn_id=prawn_id,
                filename=filename,
                pond_type=pond_type,
                manual_lengths=manual_lengths,
                manual_scales=manual_scales,
                bounding_boxes=bounding_boxes
            )
            
        except Exception as e:
            print(f"Error processing row: {e}")
            return None
    
    def process_all_measurements(self) -> List[PrawnMeasurements]:
        """
        Process all measurements from the data file.
        
        Returns:
            List of PrawnMeasurements objects
        """
        data = self.data_loader.get_data()
        measurements = []
        
        for _, row in data.iterrows():
            measurement = self.extract_measurements_from_row(row)
            if measurement:
                measurements.append(measurement)
        
        return measurements


class FilePathResolver:
    """
    Resolves file paths for predictions and ground truth data.
    
    Handles the complex logic for finding matching files across different
    directory structures that was scattered in the original code.
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        # Load test images metadata as a lookup table
        self.test_images_metadata = None
        try:
            self.test_images_metadata = pd.read_excel(self.config.METADATA_PATH)
        except Exception as e:
            print(f"Warning: Could not load test images metadata: {e}")
    
    def find_prediction_file(self, filename: str, weights_type: str) -> Optional[Path]:
        """
        Find prediction file for given filename and weights type.
        
        Args:
            filename: Base filename to search for (e.g., GX010067_33_625)
            weights_type: Type of weights used ('car', 'kalkar', 'all')
            
        Returns:
            Path to prediction file or None if not found
        """
        prediction_dir = self.config.get_prediction_path(weights_type)
        core_id = filename
        # Remove 'undistorted_' prefix if present
        if core_id.startswith('undistorted_'):
            core_id = core_id[len('undistorted_'):]
        if not prediction_dir.exists():
            return None
        for file in prediction_dir.iterdir():
            if file.is_file() and file.name.startswith(core_id) and file.name.endswith('.txt'):
                return file
        return None
    
    def find_ground_truth_file(self, filename: str) -> Optional[Path]:
        """
        Find ground truth file for given filename.
        
        Args:
            filename: Base filename to search for
            
        Returns:
            Path to ground truth file or None if not found
        """
        base_name = FilenameProcessor.extract_base_filename(filename)
        
        # Try different possible filename variations
        possible_names = [
            f"{base_name}.txt",
            f"{filename}.txt",
            f"{base_name}-jpg_gamma_jpg.txt"
        ]
        
        for name in possible_names:
            file_path = self.config.GROUND_TRUTH_PATH / name
            if file_path.exists():
                return file_path
        
        return None
    
    def find_image_file(self, filename: str, pond_type: str) -> Optional[Path]:
        """
        Find image file using test images.xlsx logic from the original code.
        Args:
            filename: Base filename to search for (e.g., undistorted_GX010069_19_191)
            pond_type: (ignored)
        Returns:
            Path to image file or None if not found
        """
        # 1. Normalise filename for search
        processed_stem = Path(filename).stem
        if processed_stem.startswith('undistorted_'):
            processed_stem_no_prefix = processed_stem[len('undistorted_'):]
        else:
            processed_stem_no_prefix = processed_stem

        # 1.a Excel-based lookup (as before) but using processed_stem_no_prefix
        if self.test_images_metadata is not None and 'file_name_new' in self.test_images_metadata.columns:
            match = self.test_images_metadata[self.test_images_metadata['file_name_new'] == processed_stem_no_prefix]
            if not match.empty and 'Label' in match.columns:
                label_val = match.iloc[0]['Label']
                if isinstance(label_val, str):
                    image_filename = label_val.split(':')[1] if ':' in label_val else label_val
                    for folder in self.config.IMAGE_PATHS.values():
                        if folder.exists():
                            for file in folder.iterdir():
                                if file.is_file() and (file.name.startswith(image_filename) or file.stem.startswith(Path(image_filename).stem)):
                                    return file

        # 2. Direct search in configured folders (match by stem w/o prefix)
        for folder in self.config.IMAGE_PATHS.values():
            if folder.exists():
                for file in folder.iterdir():
                    if file.is_file():
                        st = file.stem
                        if st.startswith(processed_stem_no_prefix) or processed_stem_no_prefix in st:
                            return file

        # 3. Not found
        return None 