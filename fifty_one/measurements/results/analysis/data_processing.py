#!/usr/bin/env python3
"""Data processing utilities for prawn measurement analysis."""

import os
import ast
import re
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm

from config import Config
from models import PoseDetection, BoundingBoxMeasurement, PrawnMeasurements


class DataLoader:
    """
    Handles loading and parsing of measurement data files.
    
    Key improvements from original code:
    - Centralized data loading logic
    - Clear separation between carapace and body data loading
    - Better error handling
    - Configurable file paths
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def load_measurement_data(self, measurement_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load measurement data based on type.
        
        Args:
            measurement_type: Either 'carapace' or 'body'
            
        Returns:
            Tuple of (filtered_df, metadata_df)
        """
        if measurement_type == 'carapace':
            return self._load_carapace_data()
        elif measurement_type == 'body':
            return self._load_body_data()
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")
    
    def _load_carapace_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load carapace measurement data."""
        filtered_df = pd.read_csv(self.config.CARAPACE_DATA_PATH)
        metadata_df = pd.read_excel(self.config.METADATA_PATH)
        return filtered_df, metadata_df
    
    def _load_body_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load body measurement data."""
        filtered_df = pd.read_excel(self.config.BODY_DATA_PATH)
        metadata_df = pd.read_excel(self.config.METADATA_PATH)
        return filtered_df, metadata_df


class YOLOParser:
    """
    Handles parsing of YOLO format files.
    
    Extracted from original parse_pose_estimation function with improvements:
    - Better error handling
    - Type hints
    - Clear documentation
    """
    
    @staticmethod
    def parse_pose_file(file_path: Path) -> List[PoseDetection]:
        """
        Parse YOLO pose estimation file.
        
        Args:
            file_path: Path to YOLO format text file
            
        Returns:
            List of PoseDetection objects
        """
        detections = []
        
        if not file_path.exists():
            return detections
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    detection = PoseDetection.from_yolo_line(line)
                    if detection:
                        detections.append(detection)
        except (IOError, OSError) as e:
            print(f"Error reading file {file_path}: {e}")
        
        return detections


class FilenameProcessor:
    """
    Handles filename processing and identifier extraction.
    
    Centralized from various extract_identifier functions in original code.
    """
    
    @staticmethod
    def extract_identifier(filename: str) -> Optional[str]:
        """
        Extract identifier pattern like 'GX010179_200_3927' from filename.
        
        Args:
            filename: The filename to process
            
        Returns:
            The extracted identifier or None if not found
        """
        # Pattern matches: GX followed by digits, underscore, digits, underscore, digits
        pattern = r'(GX\d+_\d+_\d+)'
        match = re.search(pattern, filename)
        return match.group(1) if match else None
    
    @staticmethod
    def clean_filename(filename: str) -> str:
        """
        Clean filename by removing common prefixes and extensions.
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename
        """
        # Remove 'undistorted_' prefix if present
        if 'undistorted' in filename:
            filename = filename.replace('undistorted_', '')
        
        # Remove file extension
        filename = filename.split('.')[0]
        
        return filename
    
    @staticmethod
    def create_compatible_filename(filename: str) -> str:
        """
        Create compatible filename for metadata matching.
        
        Args:
            filename: Original filename
            
        Returns:
            Compatible filename for matching
        """
        cleaned = FilenameProcessor.clean_filename(filename)
        parts = cleaned.split('_')
        
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[-1]}"
        
        return cleaned


class BoundingBoxProcessor:
    """
    Handles bounding box processing and calculations.
    
    Extracted from various bbox calculation functions in original code.
    """
    
    @staticmethod
    def calculate_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
        """Calculate bounding box area."""
        x, y, w, h = bbox
        return w * h
    
    @staticmethod
    def parse_bbox_string(bbox_str: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Parse bounding box from string representation.
        
        Args:
            bbox_str: String representation of bbox (e.g., from CSV)
            
        Returns:
            Parsed bbox tuple or None if parsing fails
        """
        try:
            bbox = ast.literal_eval(bbox_str)
            return tuple(float(coord) for coord in bbox)
        except (ValueError, SyntaxError):
            return None
    
    @staticmethod
    def normalize_bbox(bbox: Tuple[float, float, float, float], 
                      image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        """
        Normalize bounding box coordinates to [0,1] range.
        
        Args:
            bbox: Bounding box (x, y, w, h) in pixel coordinates
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Normalized bounding box
        """
        x, y, w, h = bbox
        return (x / image_width, y / image_height, w / image_width, h / image_height)
    
    @staticmethod
    def get_bbox_corners(bbox: Tuple[float, float, float, float]) -> Dict[str, Tuple[float, float]]:
        """
        Get corner coordinates of bounding box.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Dictionary with corner coordinates
        """
        x, y, w, h = bbox
        
        return {
            'top_left': (x, y),
            'top_right': (x + w, y),
            'bottom_left': (x, y + h),
            'bottom_right': (x + w, y + h)
        }
    
    @staticmethod
    def calculate_diagonal_distances(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Calculate diagonal distances of bounding box.
        
        Args:
            bbox: Bounding box (x, y, w, h)
            
        Returns:
            Tuple of (diagonal1_distance, diagonal2_distance)
        """
        import math
        
        corners = BoundingBoxProcessor.get_bbox_corners(bbox)
        
        # Calculate both diagonal distances
        diagonal1 = math.sqrt(
            (corners['bottom_right'][0] - corners['top_left'][0]) ** 2 +
            (corners['bottom_right'][1] - corners['top_left'][1]) ** 2
        )
        
        diagonal2 = math.sqrt(
            (corners['bottom_left'][0] - corners['top_right'][0]) ** 2 +
            (corners['bottom_left'][1] - corners['top_right'][1]) ** 2
        )
        
        return diagonal1, diagonal2


class PrawnDataProcessor:
    """
    Processes prawn measurement data from CSV files.
    
    This class consolidates the complex data processing logic from the original code
    into a cleaner, more maintainable structure.
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.bbox_processor = BoundingBoxProcessor()
    
    def extract_prawn_measurements(self, filtered_df: pd.DataFrame, 
                                 filename: str, measurement_type: str) -> List[PrawnMeasurements]:
        """
        Extract prawn measurements from filtered dataframe.
        
        Args:
            filtered_df: DataFrame containing measurement data
            filename: Current image filename
            measurement_type: Type of measurement ('carapace' or 'body')
            
        Returns:
            List of PrawnMeasurements objects
        """
        measurement_config = self.config.MEASUREMENT_CONFIGS[measurement_type]
        label_prefix = measurement_config.label_prefix
        
        # Find matching rows for this filename
        matching_rows = self._find_matching_rows(filtered_df, filename, label_prefix)
        
        prawn_measurements = []
        
        for _, row in matching_rows.iterrows():
            prawn_id = row['PrawnID']
            
            # Extract manual measurements
            manual_lengths = self._extract_manual_lengths(row)
            manual_scales = self._extract_manual_scales(row)
            bounding_boxes = self._extract_bounding_boxes(row)
            
            # Create measurement object
            prawn_measurement = PrawnMeasurements(
                prawn_id=prawn_id,
                filename=filename,
                pond_type="",  # Will be set later
                manual_lengths=manual_lengths,
                manual_scales=manual_scales,
                bounding_boxes=bounding_boxes
            )
            
            prawn_measurements.append(prawn_measurement)
        
        return prawn_measurements
    
    def _find_matching_rows(self, filtered_df: pd.DataFrame, 
                          filename: str, label_prefix: str) -> pd.DataFrame:
        """Find rows matching the filename and label prefix."""
        # Handle different filename formats
        identifier = FilenameProcessor.extract_identifier(filename)
        
        if identifier:
            # Try multiple matching strategies
            for pattern in [
                f'{label_prefix}:{filename}',
                f'{label_prefix}:{identifier}',
                identifier
            ]:
                matches = filtered_df[filtered_df['Label'].str.contains(pattern, na=False)]
                if not matches.empty:
                    return matches
        
        # Fallback: search within all labels
        for index, row in filtered_df.iterrows():
            if identifier and identifier in str(row['Label']):
                return filtered_df.loc[filtered_df['Label'] == row['Label']]
        
        print(f"No matching rows found for filename: {filename}")
        return pd.DataFrame()
    
    def _extract_manual_lengths(self, row: pd.Series) -> List[float]:
        """Extract Length_1, Length_2, Length_3 from row."""
        lengths = []
        for i in range(1, 4):
            length_key = f'Length_{i}'
            if length_key in row and pd.notna(row[length_key]):
                lengths.append(abs(float(row[length_key])))
            else:
                lengths.append(None)
        return lengths
    
    def _extract_manual_scales(self, row: pd.Series) -> List[float]:
        """Extract Scale_1, Scale_2, Scale_3 from row."""
        scales = []
        for i in range(1, 4):
            scale_key = f'Scale_{i}'
            if scale_key in row and pd.notna(row[scale_key]):
                scales.append(abs(float(row[scale_key])))
            else:
                scales.append(None)
        return scales
    
    def _extract_bounding_boxes(self, row: pd.Series) -> List[BoundingBoxMeasurement]:
        """Extract and process bounding boxes from row."""
        bounding_boxes = []
        
        for i in range(1, 4):
            bbox_key = f'BoundingBox_{i}'
            length_key = f'Length_{i}'
            
            if (bbox_key in row and pd.notna(row[bbox_key]) and 
                length_key in row and pd.notna(row[length_key])):
                
                bbox = self.bbox_processor.parse_bbox_string(str(row[bbox_key]))
                length_mm = abs(float(row[length_key]))
                
                if bbox:
                    bbox_measurement = BoundingBoxMeasurement.from_bbox_and_length(bbox, length_mm)
                    bounding_boxes.append(bbox_measurement)
        
        return bounding_boxes


class FilePathResolver:
    """
    Resolves file paths for predictions and ground truth files.
    
    Handles the complex file matching logic from the original code.
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def find_prediction_file(self, identifier: str, prediction_folder: Path) -> Optional[Path]:
        """
        Find prediction file matching the identifier.
        
        Args:
            identifier: Image identifier to match
            prediction_folder: Folder containing prediction files
            
        Returns:
            Path to matching prediction file or None
        """
        if not prediction_folder.exists():
            return None
        
        for pred_file in prediction_folder.iterdir():
            if pred_file.suffix == '.txt' and identifier in pred_file.name:
                return pred_file
        
        return None
    
    def find_ground_truth_file(self, identifier: str, ground_truth_files: List[Path]) -> Optional[Path]:
        """
        Find ground truth file matching the identifier.
        
        Args:
            identifier: Image identifier to match
            ground_truth_files: List of ground truth file paths
            
        Returns:
            Path to matching ground truth file or None
        """
        for gt_file in ground_truth_files:
            gt_identifier = FilenameProcessor.extract_identifier(gt_file.name)
            if gt_identifier == identifier:
                return gt_file
        
        return None
    
    def get_image_paths(self, pond_type: str) -> List[Path]:
        """
        Get image paths for specified pond type.
        
        Args:
            pond_type: Type of pond (right, left, car)
            
        Returns:
            List of image file paths
        """
        folder_path = self.config.IMAGE_PATHS.get(pond_type)
        if not folder_path or not folder_path.exists():
            return []
        
        image_extensions = {'.png', '.jpg', '.jpeg'}
        image_paths = []
        
        for file_path in folder_path.iterdir():
            if file_path.suffix.lower() in image_extensions:
                image_paths.append(file_path)
        
        return image_paths
    
    def get_ground_truth_paths(self) -> List[Path]:
        """Get all ground truth file paths."""
        if not self.config.GROUND_TRUTH_PATH.exists():
            return []
        
        return [f for f in self.config.GROUND_TRUTH_PATH.iterdir() if f.suffix == '.txt'] 