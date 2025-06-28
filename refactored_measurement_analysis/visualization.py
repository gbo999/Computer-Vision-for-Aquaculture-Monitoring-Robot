#!/usr/bin/env python3
"""FiftyOne visualization and dataset management utilities."""

import fiftyone as fo
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import socket

from .models import PrawnMeasurements, MeasurementResult, PoseDetection
from .config import Config


class PortManager:
    """
    Manages FiftyOne port allocation to avoid conflicts.
    
    Extracted from the original port management logic to provide
    a clean interface for finding available ports.
    """
    
    @staticmethod
    def find_available_port(start_port: int = 5150, end_port: int = 5190) -> int:
        """
        Find an available port in the specified range.
        
        Args:
            start_port: Starting port number
            end_port: Ending port number
            
        Returns:
            Available port number
            
        Raises:
            RuntimeError: If no available port found
        """
        for port in range(start_port, end_port + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                continue
        
        raise RuntimeError(f"No available port found in range {start_port}-{end_port}")


class FiftyOneDatasetManager:
    """
    Manages FiftyOne dataset creation and persistence.
    
    This class handles the dataset lifecycle that was previously
    mixed with analysis logic in the original code.
    """
    
    def __init__(self, dataset_name: str, port: Optional[int] = None):
        """
        Initialize dataset manager.
        
        Args:
            dataset_name: Name for the FiftyOne dataset
            port: Optional port number for FiftyOne app
        """
        self.dataset_name = dataset_name
        self.port = port or PortManager.find_available_port()
        self.dataset = None
    
    def create_or_load_dataset(self, overwrite: bool = False) -> fo.Dataset:
        """
        Create new dataset or load existing one.
        
        Args:
            overwrite: Whether to overwrite existing dataset
            
        Returns:
            FiftyOne dataset
        """
        if overwrite and self.dataset_name in fo.list_datasets():
            fo.delete_dataset(self.dataset_name)
        
        if self.dataset_name in fo.list_datasets():
            self.dataset = fo.load_dataset(self.dataset_name)
            print(f"Loaded existing dataset: {self.dataset_name}")
        else:
            self.dataset = fo.Dataset(self.dataset_name)
            print(f"Created new dataset: {self.dataset_name}")
        
        return self.dataset
    
    def launch_app(self, **kwargs):
        """
        Launch FiftyOne app for the dataset.
        
        Args:
            **kwargs: Additional arguments for fo.launch_app()
            
        Returns:
            FiftyOne app instance
        """
        if not self.dataset:
            raise RuntimeError("Dataset not created. Call create_or_load_dataset() first.")
        
        app_kwargs = {'port': self.port}
        app_kwargs.update(kwargs)
        
        return fo.launch_app(self.dataset, **app_kwargs)
    
    def persist_dataset(self):
        """Persist the dataset to disk."""
        if self.dataset:
            self.dataset.persistent = True
            print(f"Dataset {self.dataset_name} persisted to disk")


class FiftyOneSampleProcessor:
    """
    Processes individual samples for FiftyOne visualization.
    
    Extracted from sample processing logic that was embedded
    in the main analysis functions.
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def create_sample_from_measurement(self, measurement: PrawnMeasurements, 
                                     image_path: Optional[Path] = None) -> Optional[fo.Sample]:
        """
        Create FiftyOne sample from prawn measurement data.
        
        Args:
            measurement: PrawnMeasurements object
            image_path: Optional path to image file
            
        Returns:
            FiftyOne sample or None if image not found
        """
        if not image_path:
            from .data_processing import FilePathResolver
            resolver = FilePathResolver(self.config)
            original_filename = measurement.filename
            cleaned_filename = original_filename
            if cleaned_filename.startswith('undistorted_'):
                cleaned_filename = cleaned_filename[len('undistorted_'):]
            image_path = resolver.find_image_file(original_filename, measurement.pond_type)
        
        if not image_path or not image_path.exists():
            print(f"Image not found for original: {original_filename} | cleaned: {cleaned_filename}")
            return None
        
        # Create basic sample
        sample = fo.Sample(filepath=str(image_path))
        
        # Add metadata
        sample["prawn_id"] = measurement.prawn_id
        sample["pond_type"] = measurement.pond_type
        sample["filename"] = measurement.filename
        
        # Add manual measurements
        if measurement.manual_lengths:
            sample["manual_min_length"] = measurement.min_length
            sample["manual_max_length"] = measurement.max_length
            sample["manual_median_length"] = measurement.median_length
        
        # Add predicted measurements
        if measurement.predicted_measurement:
            sample["predicted_length_mm"] = measurement.predicted_measurement.distance_mm
            sample["predicted_length_px"] = measurement.predicted_measurement.distance_pixels
            sample["measurement_angle"] = measurement.predicted_measurement.angle_deg
        
        # Add ground truth measurements
        if measurement.ground_truth_measurement:
            sample["ground_truth_length_mm"] = measurement.ground_truth_measurement.distance_mm
            sample["ground_truth_length_px"] = measurement.ground_truth_measurement.distance_pixels
        
        # Add error metrics
        errors = measurement.calculate_errors()
        for error_name, error_value in errors.items():
            sample[f"error_{error_name}"] = error_value
        
        return sample
    
    def add_pose_detections(self, sample: fo.Sample, 
                           detections: List[PoseDetection],
                           field_name: str = "pose_detections") -> fo.Sample:
        """
        Add pose detection data to sample.
        
        Args:
            sample: FiftyOne sample
            detections: List of pose detections
            field_name: Field name for detections
            
        Returns:
            Updated sample
        """
        fo_detections = []
        
        for detection in detections:
            # Convert bbox to FiftyOne format
            x, y, w, h = detection.bbox
            bbox = [x - w/2, y - h/2, w, h]  # Convert center format to top-left
            
            # Create keypoints
            keypoints = []
            for kp_x, kp_y, confidence in detection.keypoints:
                keypoints.extend([kp_x, kp_y])
            
            fo_detection = fo.Detection(
                label=f"prawn_{detection.class_id}",
                bounding_box=bbox,
                confidence=1.0,  # Overall detection confidence
            )
            
            # Add keypoints as custom field
            fo_detection["keypoints"] = keypoints
            fo_detections.append(fo_detection)
        
        sample[field_name] = fo.Detections(detections=fo_detections)
        return sample


class VisualizationCreator:
    """
    Creates visual annotations for FiftyOne display.
    
    Handles the creation of polylines, measurement annotations,
    and other visual elements that were mixed in with analysis code.
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def create_measurement_polylines(self, measurement: PrawnMeasurements, 
                                   detection: Optional[PoseDetection] = None) -> List[fo.Polyline]:
        """
        Create polylines showing measurement lines.
        
        Args:
            measurement: PrawnMeasurements object
            detection: Optional pose detection for keypoint lines
            
        Returns:
            List of FiftyOne polylines
        """
        polylines = []
        
        if not detection or not measurement.predicted_measurement:
            return polylines
        
        # Get measurement configuration
        measurement_config = self.config.MEASUREMENT_CONFIGS.get(
            'carapace' if 'carapace' in measurement.filename.lower() else 'body'
        )
        
        if not measurement_config:
            return polylines
        
        # Create measurement line based on keypoint classes
        if measurement_config.keypoint_classes == ["start-carapace", "eyes"]:
            # Carapace measurement: line between start-carapace and eyes
            if len(detection.keypoints) >= 2:
                start_point = detection.keypoints[0]  # start-carapace
                end_point = detection.keypoints[1]    # eyes
                
                polyline = fo.Polyline(
                    label=f"{measurement_config.label_prefix}_measurement",
                    points=[
                        [(start_point[0], start_point[1]), (end_point[0], end_point[1])]
                    ],
                    filled=False
                )
                polyline["length_mm"] = measurement.predicted_measurement.distance_mm
                polyline["length_px"] = measurement.predicted_measurement.distance_pixels
                polylines.append(polyline)
        
        elif measurement_config.keypoint_classes == ["tail", "rostrum"]:
            # Body measurement: line between tail and rostrum
            if len(detection.keypoints) >= 4:
                tail_point = detection.keypoints[3]   # tail
                rostrum_point = detection.keypoints[2]  # rostrum
                
                polyline = fo.Polyline(
                    label=f"{measurement_config.label_prefix}_measurement",
                    points=[
                        [(tail_point[0], tail_point[1]), (rostrum_point[0], rostrum_point[1])]
                    ],
                    filled=False
                )
                polyline["length_mm"] = measurement.predicted_measurement.distance_mm
                polyline["length_px"] = measurement.predicted_measurement.distance_pixels
                polylines.append(polyline)
        
        return polylines
    
    def create_skeleton_polylines(self, detection: PoseDetection, 
                                 measurement_type: str) -> List[fo.Polyline]:
        """
        Create skeleton polylines connecting keypoints.
        
        Args:
            detection: Pose detection with keypoints
            measurement_type: Type of measurement ('carapace' or 'body')
            
        Returns:
            List of skeleton polylines
        """
        polylines = []
        
        measurement_config = self.config.MEASUREMENT_CONFIGS.get(measurement_type)
        if not measurement_config or len(detection.keypoints) < 4:
            return polylines
        
        # Create skeleton edges
        for edge in measurement_config.skeleton_edges:
            if len(edge) == 2 and all(idx < len(detection.keypoints) for idx in edge):
                start_kp = detection.keypoints[edge[0]]
                end_kp = detection.keypoints[edge[1]]
                
                # Only create edge if both keypoints have sufficient confidence
                if start_kp[2] > 0.5 and end_kp[2] > 0.5:
                    polyline = fo.Polyline(
                        label=f"skeleton_{edge[0]}_{edge[1]}",
                        points=[
                            [(start_kp[0], start_kp[1]), (end_kp[0], end_kp[1])]
                        ],
                        filled=False
                    )
                    polylines.append(polyline)
        
        return polylines


class ErrorTagManager:
    """
    Manages error-based tagging of samples for easy filtering.
    
    Provides functionality to tag samples based on error thresholds
    that was previously embedded in the analysis workflow.
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def tag_sample_by_error(self, sample: fo.Sample, 
                           error_percentage: float) -> fo.Sample:
        """
        Tag sample based on error percentage.
        
        Args:
            sample: FiftyOne sample
            error_percentage: Error percentage value
            
        Returns:
            Updated sample with error tags
        """
        tags = []
        
        # Add error level tags
        if error_percentage > self.config.ERROR_THRESHOLDS['high']:
            tags.append('high_error')
        elif error_percentage > self.config.ERROR_THRESHOLDS['medium']:
            tags.append('medium_error')
        elif error_percentage > self.config.ERROR_THRESHOLDS['low']:
            tags.append('low_error')
        elif error_percentage > self.config.ERROR_THRESHOLDS['very_low']:
            tags.append('very_low_error')
        else:
            tags.append('excellent')
        
        # Add precision tags
        if error_percentage < 5:
            tags.append('high_precision')
        elif error_percentage < 15:
            tags.append('medium_precision')
        else:
            tags.append('low_precision')
        
        # Update sample tags
        existing_tags = sample.tags or []
        sample.tags = list(set(existing_tags + tags))
        
        return sample
    
    def tag_dataset_by_errors(self, dataset: fo.Dataset, 
                             error_field: str = "error_mpe_median") -> fo.Dataset:
        """
        Tag all samples in dataset based on error values.
        
        Args:
            dataset: FiftyOne dataset
            error_field: Field name containing error values
            
        Returns:
            Updated dataset
        """
        for sample in dataset:
            if error_field in sample:
                error_value = sample[error_field]
                if error_value is not None:
                    self.tag_sample_by_error(sample, error_value)
                    sample.save()
        
        return dataset 