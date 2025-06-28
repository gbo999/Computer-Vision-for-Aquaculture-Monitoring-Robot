#!/usr/bin/env python3
"""Data models and measurement calculation classes."""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class PoseDetection:
    """Represents a YOLO pose detection."""
    class_id: int
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    keypoints: List[Tuple[float, float, float]]  # x, y, confidence for each keypoint
    
    @classmethod
    def from_yolo_line(cls, line: str) -> Optional['PoseDetection']:
        """Parse YOLO format line into PoseDetection."""
        try:
            parts = list(map(float, line.strip().split()))
            if len(parts) == 17:  # 1 class + 4 bbox + 4*3 keypoints
                class_id = int(parts[0])
                bbox = tuple(parts[1:5])
                keypoints = []
                for i in range(5, len(parts), 3):
                    keypoints.append((parts[i], parts[i+1], parts[i+2]))
                return cls(class_id, bbox, keypoints)
        except (ValueError, IndexError):
            pass
        return None


@dataclass
class MeasurementResult:
    """Represents measurement calculation results."""
    distance_pixels: float
    distance_mm: float
    angle_deg: float
    combined_scale: float
    focal_length_mm: Optional[float] = None


class ObjectLengthMeasurer:
    """
    Calculates real-world measurements from pixel distances using camera parameters.
    
    This class handles conversion from pixel measurements to millimeters using:
    1. Camera field of view (FOV) method
    2. Focal length method
    
    Key differences from original implementation:
    - Extracted from data_loader.py into separate class
    - Added clear documentation for each method
    - Simplified angle normalization logic
    """
    
    def __init__(self, image_width: int, image_height: int, 
                 horizontal_fov: float, vertical_fov: float, distance_mm: float):
        """
        Initialize the measurer with camera parameters.
        
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels  
            horizontal_fov: Horizontal field of view in degrees
            vertical_fov: Vertical field of view in degrees
            distance_mm: Distance from camera to subject in millimeters
        """
        self.image_width = image_width
        self.image_height = image_height
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.distance_mm = distance_mm
        self.scale_x, self.scale_y = self._calculate_scaling_factors()
    
    def _calculate_scaling_factors(self) -> Tuple[float, float]:
        """
        Calculate scaling factors (mm per pixel) based on camera FOV and distance.
        
        Uses trigonometry: scale = 2 * distance * tan(fov/2) / image_dimension
        
        Returns:
            Tuple of (scale_x, scale_y) in mm per pixel
        """
        fov_x_rad = math.radians(self.horizontal_fov)
        fov_y_rad = math.radians(self.vertical_fov)
        
        scale_x = (2 * self.distance_mm * math.tan(fov_x_rad / 2)) / self.image_width
        scale_y = (2 * self.distance_mm * math.tan(fov_y_rad / 2)) / self.image_height
        
        return scale_x, scale_y
    
    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to range [0, 90].
        
        Args:
            angle: Angle in degrees
            
        Returns:
            Normalized angle in degrees
        """
        theta_norm = min(abs(angle % 180), 180 - abs(angle % 180))
        return theta_norm
    
    def compute_length_from_distance(self, pixel_distance: float, angle_deg: float) -> MeasurementResult:
        """
        Compute real-world length from pixel distance and angle.
        
        Args:
            pixel_distance: Distance in pixels
            angle_deg: Angle of measurement line in degrees
            
        Returns:
            MeasurementResult with distance in mm and scaling information
        """
        angle_rad = math.radians(angle_deg)
        combined_scale = math.sqrt(
            (self.scale_x * math.cos(angle_rad)) ** 2 + 
            (self.scale_y * math.sin(angle_rad)) ** 2
        )
        length_mm = pixel_distance * combined_scale
        
        return MeasurementResult(
            distance_pixels=pixel_distance,
            distance_mm=length_mm,
            angle_deg=angle_deg,
            combined_scale=combined_scale
        )
    
    def compute_length_between_points(self, point1: Tuple[float, float], 
                                    point2: Tuple[float, float]) -> MeasurementResult:
        """
        Compute real-world distance between two points.
        
        Args:
            point1: First point (x, y) in pixels
            point2: Second point (x, y) in pixels
            
        Returns:
            MeasurementResult with comprehensive measurement data
        """
        # Calculate pixel distance and angle
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        distance_px = math.sqrt(delta_x ** 2 + delta_y ** 2)
        
        # Calculate angle
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
        normalized_angle = self._normalize_angle(angle_deg)
        
        # Compute real-world distance
        return self.compute_length_from_distance(distance_px, normalized_angle)


class FocalLengthMeasurer:
    """
    Alternative measurement method using focal length calculations.
    
    This class provides the focal length method for comparison with FOV method.
    Extracted from calculate_real_width function in original code.
    """
    
    @staticmethod
    def calculate_real_width(focal_length: float, height_mm: float, 
                           pixel_distance: float, pixel_size: float) -> float:
        """
        Calculate real-world width using focal length method.
        
        Args:
            focal_length: Camera focal length in mm
            height_mm: Distance from camera to subject in mm
            pixel_distance: Distance in pixels
            pixel_size: Size of one pixel in mm
            
        Returns:
            Real-world distance in mm
        """
        return (pixel_distance * pixel_size * height_mm) / focal_length


@dataclass
class BoundingBoxMeasurement:
    """Represents measurements for a single bounding box."""
    bbox: Tuple[float, float, float, float]  # x, y, w, h
    length_mm: float
    diagonal_1_px: float
    diagonal_2_px: float
    area: float
    
    @classmethod
    def from_bbox_and_length(cls, bbox: Tuple[float, float, float, float], 
                           length_mm: float) -> 'BoundingBoxMeasurement':
        """Create measurement from bounding box and length."""
        x, y, w, h = bbox
        
        # Calculate diagonal distances
        diagonal_1_px = math.sqrt(w ** 2 + h ** 2)  # Top-left to bottom-right
        diagonal_2_px = math.sqrt(w ** 2 + h ** 2)  # Top-right to bottom-left (same)
        
        area = w * h
        
        return cls(bbox, length_mm, diagonal_1_px, diagonal_2_px, area)


@dataclass
class PrawnMeasurements:
    """Complete measurement data for a single prawn."""
    prawn_id: str
    filename: str
    pond_type: str
    
    # Manual measurements from ImageJ
    manual_lengths: List[float]  # Length_1, Length_2, Length_3
    manual_scales: List[float]   # Scale_1, Scale_2, Scale_3
    bounding_boxes: List[BoundingBoxMeasurement]
    
    # Predicted measurements
    predicted_measurement: Optional[MeasurementResult] = None
    ground_truth_measurement: Optional[MeasurementResult] = None
    focal_length_measurement: Optional[float] = None
    
    # Calculated values
    min_length: Optional[float] = None
    max_length: Optional[float] = None
    median_length: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived values after initialization."""
        if self.manual_lengths:
            abs_lengths = [abs(length) for length in self.manual_lengths if length is not None]
            if abs_lengths:
                self.min_length = min(abs_lengths)
                self.max_length = max(abs_lengths)
                # Calculate median as middle value when sorted
                sorted_lengths = sorted(abs_lengths)
                if len(sorted_lengths) == 3:
                    self.median_length = sorted_lengths[1]
                elif len(sorted_lengths) > 0:
                    self.median_length = sorted_lengths[len(sorted_lengths) // 2]
    
    def calculate_errors(self) -> dict:
        """Calculate various error metrics."""
        errors = {}
        
        if not (self.predicted_measurement and self.min_length and self.max_length and self.median_length):
            return errors
        
        pred_mm = self.predicted_measurement.distance_mm
        
        # Percentage errors
        errors['mpe_min'] = abs(pred_mm - self.min_length) / self.min_length * 100
        errors['mpe_max'] = abs(pred_mm - self.max_length) / self.max_length * 100
        errors['mpe_median'] = abs(pred_mm - self.median_length) / self.median_length * 100
        
        # Absolute errors
        errors['abs_error_min'] = abs(pred_mm - self.min_length)
        errors['abs_error_max'] = abs(pred_mm - self.max_length)
        errors['abs_error_median'] = abs(pred_mm - self.median_length)
        
        # Ground truth errors if available
        if self.ground_truth_measurement:
            gt_mm = self.ground_truth_measurement.distance_mm
            errors['gt_error_min'] = abs(gt_mm - self.min_length)
            errors['gt_error_max'] = abs(gt_mm - self.max_length)
            errors['gt_error_median'] = abs(gt_mm - self.median_length)
            
            errors['gt_mpe_min'] = errors['gt_error_min'] / self.min_length * 100
            errors['gt_mpe_max'] = errors['gt_error_max'] / self.max_length * 100
            errors['gt_mpe_median'] = errors['gt_error_median'] / self.median_length * 100
        
        # Focal length errors if available
        if self.focal_length_measurement:
            focal_mm = self.focal_length_measurement
            errors['focal_error_min'] = abs(focal_mm - self.min_length)
            errors['focal_error_max'] = abs(focal_mm - self.max_length)
            errors['focal_error_median'] = abs(focal_mm - self.median_length)
            
            errors['focal_mpe_min'] = errors['focal_error_min'] / self.min_length * 100
            errors['focal_mpe_max'] = errors['focal_error_max'] / self.max_length * 100
            errors['focal_mpe_median'] = errors['focal_error_median'] / self.median_length * 100
        
        return errors
    
    def get_min_error_percentage(self) -> float:
        """Get the minimum error percentage across all comparison methods."""
        errors = self.calculate_errors()
        error_values = [
            errors.get('mpe_min', float('inf')),
            errors.get('mpe_max', float('inf')),
            errors.get('mpe_median', float('inf'))
        ]
        return min(error_values) if any(e != float('inf') for e in error_values) else 0.0 