#!/usr/bin/env python3
"""FiftyOne visualization and dataset management."""

import fiftyone as fo
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from config import Config
from models import PoseDetection, PrawnMeasurements, MeasurementResult
from data_processing import FilenameProcessor, BoundingBoxProcessor


class FiftyOneDatasetManager:
    """
    Manages FiftyOne dataset creation and persistence.
    
    Extracted from original create_dataset functions with improvements:
    - Unified dataset creation logic
    - Better error handling
    - Configurable persistence
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def create_or_load_dataset(self, measurement_type: str, weights_type: str) -> Tuple[fo.Dataset, bool]:
        """
        Create new dataset or load existing one.
        
        Args:
            measurement_type: Type of measurement ('carapace' or 'body')
            weights_type: Type of weights used
            
        Returns:
            Tuple of (dataset, exists_flag)
        """
        dataset_name = self.config.get_dataset_name(measurement_type, weights_type)
        export_path = self.config.THESIS_EXPORT_PATH / f"{measurement_type}_{weights_type}"
        
        # Check if dataset already exists
        if export_path.exists():
            try:
                dataset = fo.load_dataset(dataset_name)
                if dataset:
                    print(f"Dataset {dataset_name} loaded successfully")
                    return dataset, True
            except Exception as e:
                print(f"Failed to load existing dataset: {e}")
        
        # Create new dataset
        print(f"Creating new dataset: {dataset_name}")
        dataset = fo.Dataset(dataset_name, overwrite=True, persistent=True)
        
        # Set up skeleton configuration
        measurement_config = self.config.MEASUREMENT_CONFIGS[measurement_type]
        dataset.default_skeleton = fo.KeypointSkeleton(
            labels=measurement_config.skeleton_labels,
            edges=measurement_config.skeleton_edges
        )
        
        return dataset, False
    
    def save_and_export_dataset(self, dataset: fo.Dataset, measurement_type: str, weights_type: str):
        """
        Save dataset and export if needed.
        
        Args:
            dataset: FiftyOne dataset to save
            measurement_type: Type of measurement
            weights_type: Type of weights used
        """
        # Make dataset persistent
        dataset.persistent = True
        dataset.save()
        
        # Export if path doesn't exist
        export_path = self.config.THESIS_EXPORT_PATH / f"{measurement_type}_{weights_type}"
        if not export_path.exists():
            dataset.export(
                export_dir=str(export_path),
                dataset_type=fo.types.FiftyOneDataset,
                export_media=True
            )
            print(f"Dataset exported to {export_path}")


class FiftyOneSampleProcessor:
    """
    Processes individual samples for FiftyOne visualization.
    
    Extracted from original add_metadata and process_detection functions with improvements:
    - Cleaner separation of concerns
    - Better error handling
    - Reduced code duplication
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.bbox_processor = BoundingBoxProcessor()
    
    def create_sample_with_detections(self, image_path: Path, 
                                    predictions: List[PoseDetection],
                                    ground_truths: List[PoseDetection],
                                    pond_type: str) -> fo.Sample:
        """
        Create FiftyOne sample with predictions and ground truth.
        
        Args:
            image_path: Path to image file
            predictions: List of predicted pose detections
            ground_truths: List of ground truth pose detections
            pond_type: Type of pond (right, left, car)
            
        Returns:
            Configured FiftyOne sample
        """
        sample = fo.Sample(filepath=str(image_path))
        
        # Convert detections to FiftyOne format
        pred_keypoints, pred_detections = self._process_pose_detections(predictions, is_ground_truth=False)
        gt_keypoints, gt_detections = self._process_pose_detections(ground_truths, is_ground_truth=True)
        
        # Add detections to sample
        sample["ground_truth"] = fo.Detections(detections=gt_detections)
        sample["detections_predictions"] = fo.Detections(detections=pred_detections)
        sample["keypoints"] = fo.Keypoints(keypoints=pred_keypoints)
        sample["keypoints_truth"] = fo.Keypoints(keypoints=gt_keypoints)
        
        # Add pond type tag
        sample.tags.append(pond_type)
        
        return sample
    
    def add_metadata_to_sample(self, sample: fo.Sample, filename: str, 
                             metadata_df, measurement_type: str):
        """
        Add camera metadata to sample.
        
        Args:
            sample: FiftyOne sample to update
            filename: Image filename for metadata lookup
            metadata_df: DataFrame containing camera metadata
            measurement_type: Type of measurement being processed
        """
        # Create compatible filename for metadata matching
        compatible_name = FilenameProcessor.create_compatible_filename(filename)
        
        # Find matching metadata
        metadata_row = metadata_df[metadata_df['file_name_new'] == compatible_name]
        if not metadata_row.empty:
            metadata = metadata_row.iloc[0].to_dict()
            for key, value in metadata.items():
                if key != 'file name':
                    sample[key] = value
        else:
            print(f"No metadata found for {compatible_name}")
    
    def _process_pose_detections(self, poses: List[PoseDetection], 
                               is_ground_truth: bool = False) -> Tuple[List[fo.Keypoint], List[fo.Detection]]:
        """
        Process pose detections into FiftyOne format.
        
        Args:
            poses: List of pose detections
            is_ground_truth: Whether these are ground truth annotations
            
        Returns:
            Tuple of (keypoints_list, detections_list)
        """
        keypoints_list = []
        detections = []
        
        for pose in poses:
            if len(pose.keypoints) == 4:  # Expected: start_carapace, eyes, rostrum, tail
                # Create keypoint object
                keypoint_points = [[kp[0], kp[1]] for kp in pose.keypoints]
                keypoint = fo.Keypoint(points=keypoint_points)
                keypoints_list.append(keypoint)
                
                # Create keypoints dictionary
                keypoints_dict = {
                    'start_carapace': keypoint_points[0],
                    'eyes': keypoint_points[1],
                    'rostrum': keypoint_points[2],
                    'tail': keypoint_points[3],
                    'keypoint_ID': keypoint.id
                }
                
                # Adjust bounding box format
                x, y, w, h = pose.bbox
                bbox_normalized = [x - w/2, y - h/2, w, h]  # Convert center format to corner format
                
                # Create detection
                label = "prawn_truth" if is_ground_truth else "prawn"
                detection = fo.Detection(
                    label=label,
                    bounding_box=bbox_normalized,
                    attributes={'keypoints': keypoints_dict}
                )
                detections.append(detection)
        
        return keypoints_list, detections


class VisualizationCreator:
    """
    Creates visualizations and polylines for FiftyOne samples.
    
    Extracted from diagonal line creation functions with improvements:
    - Cleaner organization
    - Reduced code duplication
    - More flexible visualization options
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        self.bbox_processor = BoundingBoxProcessor()
    
    def add_measurement_visualizations(self, sample: fo.Sample, 
                                     prawn_measurements: List[PrawnMeasurements],
                                     measurement_type: str):
        """
        Add measurement visualizations to sample.
        
        Args:
            sample: FiftyOne sample to update
            prawn_measurements: List of prawn measurement data
            measurement_type: Type of measurement being visualized
        """
        if measurement_type == 'body':
            self._add_body_visualizations(sample, prawn_measurements)
        else:
            self._add_carapace_visualizations(sample, prawn_measurements)
    
    def _add_body_visualizations(self, sample: fo.Sample, prawn_measurements: List[PrawnMeasurements]):
        """Add body measurement visualizations."""
        diagonal_lines = {'min_1': [], 'min_2': [], 'mid_1': [], 'mid_2': [], 'max_1': [], 'max_2': []}
        
        for prawn_measurement in prawn_measurements:
            if len(prawn_measurement.bounding_boxes) >= 3:
                # Sort by area
                sorted_boxes = sorted(prawn_measurement.bounding_boxes, key=lambda x: x.area)
                
                min_box, mid_box, max_box = sorted_boxes[0], sorted_boxes[1], sorted_boxes[2]
                
                # Create diagonal visualizations for each box
                self._add_box_diagonals(diagonal_lines, 'min', min_box, self.config.CAMERA.image_width, self.config.CAMERA.image_height)
                self._add_box_diagonals(diagonal_lines, 'mid', mid_box, self.config.CAMERA.image_width, self.config.CAMERA.image_height)
                self._add_box_diagonals(diagonal_lines, 'max', max_box, self.config.CAMERA.image_width, self.config.CAMERA.image_height)
        
        # Add polylines to sample
        for key, polylines in diagonal_lines.items():
            if polylines:
                sample[f"{key}_diagonal_line"] = fo.Polylines(polylines=polylines)
    
    def _add_carapace_visualizations(self, sample: fo.Sample, prawn_measurements: List[PrawnMeasurements]):
        """Add carapace measurement visualizations with length annotations."""
        diagonal_lines = {'min_1': [], 'min_2': [], 'mid_1': [], 'mid_2': [], 'max_1': [], 'max_2': []}
        
        for prawn_measurement in prawn_measurements:
            if len(prawn_measurement.bounding_boxes) >= 3:
                # Sort by area
                sorted_boxes = sorted(prawn_measurement.bounding_boxes, key=lambda x: x.area)
                
                min_box, mid_box, max_box = sorted_boxes[0], sorted_boxes[1], sorted_boxes[2]
                
                # Create diagonal visualizations with length labels
                self._add_labeled_box_diagonals(diagonal_lines, 'min', min_box, self.config.CAMERA.image_width, self.config.CAMERA.image_height)
                self._add_labeled_box_diagonals(diagonal_lines, 'mid', mid_box, self.config.CAMERA.image_width, self.config.CAMERA.image_height)
                self._add_labeled_box_diagonals(diagonal_lines, 'max', max_box, self.config.CAMERA.image_width, self.config.CAMERA.image_height)
        
        # Add polylines to sample
        for key, polylines in diagonal_lines.items():
            if polylines:
                sample[f"{key}_diagonal_line"] = fo.Polylines(polylines=polylines)
    
    def _add_box_diagonals(self, diagonal_lines: Dict, size_key: str, box_measurement, image_width: int, image_height: int):
        """Add diagonal lines for a bounding box."""
        # Normalize bounding box
        normalized_bbox = self.bbox_processor.normalize_bbox(box_measurement.bbox, image_width, image_height)
        corners = self.bbox_processor.get_bbox_corners(normalized_bbox)
        
        # Create diagonal polylines
        diagonal1 = [corners['top_left'], corners['bottom_right']]
        diagonal2 = [corners['top_right'], corners['bottom_left']]
        
        colors = {'min': ('blue', 'green'), 'mid': ('blue', 'green'), 'max': ('red', 'yellow')}
        color1, color2 = colors.get(size_key, ('blue', 'green'))
        
        polyline1 = fo.Polyline(
            label=f"{size_key} diagonal - {box_measurement.length_mm:.2f}mm",
            points=[diagonal1],
            closed=False,
            filled=False,
            line_color=color1,
            thickness=2
        )
        
        polyline2 = fo.Polyline(
            label=f"{size_key} diagonal - {box_measurement.length_mm:.2f}mm",
            points=[diagonal2],
            closed=False,
            filled=False,
            line_color=color2,
            thickness=2
        )
        
        diagonal_lines[f'{size_key}_1'].append(polyline1)
        diagonal_lines[f'{size_key}_2'].append(polyline2)
    
    def _add_labeled_box_diagonals(self, diagonal_lines: Dict, size_key: str, box_measurement, image_width: int, image_height: int):
        """Add diagonal lines with length labels for a bounding box."""
        # Similar to _add_box_diagonals but with detailed labeling
        normalized_bbox = self.bbox_processor.normalize_bbox(box_measurement.bbox, image_width, image_height)
        corners = self.bbox_processor.get_bbox_corners(normalized_bbox)
        
        diagonal1 = [corners['top_left'], corners['bottom_right']]
        diagonal2 = [corners['top_right'], corners['bottom_left']]
        
        colors = {'min': ('blue', 'green'), 'mid': ('blue', 'green'), 'max': ('red', 'yellow')}
        color1, color2 = colors.get(size_key, ('blue', 'green'))
        
        polyline1 = fo.Polyline(
            label=f"{size_key.title()} diagonal 1 - Length: {box_measurement.length_mm:.2f}mm",
            points=[diagonal1],
            closed=False,
            filled=False,
            line_color=color1,
            thickness=2
        )
        polyline1.label = f'{box_measurement.length_mm:.2f}mm'
        
        polyline2 = fo.Polyline(
            label=f"{size_key.title()} diagonal 2 - Length: {box_measurement.length_mm:.2f}mm",
            points=[diagonal2],
            closed=False,
            filled=False,
            line_color=color2,
            thickness=2
        )
        polyline2.label = f'{box_measurement.length_mm:.2f}mm'
        
        diagonal_lines[f'{size_key}_1'].append(polyline1)
        diagonal_lines[f'{size_key}_2'].append(polyline2)


class ErrorTagManager:
    """
    Manages error-based tagging of samples.
    
    Extracted from error tagging logic with improvements:
    - Configurable thresholds
    - Clear tag management
    - Extensible for different error types
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def add_error_tags(self, sample: fo.Sample, prawn_measurements: List[PrawnMeasurements]):
        """
        Add error-based tags to sample based on measurement accuracy.
        
        Args:
            sample: FiftyOne sample to tag
            prawn_measurements: List of prawn measurements with error calculations
        """
        if not prawn_measurements:
            return
        
        # Calculate minimum error across all prawns in sample
        min_errors = []
        for prawn_measurement in prawn_measurements:
            min_error = prawn_measurement.get_min_error_percentage()
            if min_error > 0:
                min_errors.append(min_error)
        
        if not min_errors:
            return
        
        overall_min_error = min(min_errors)
        
        # Add tags based on error thresholds
        thresholds = self.config.ERROR_THRESHOLDS
        
        if overall_min_error > thresholds['high']:
            self._add_unique_tag(sample, "MPE_fov>50")
        elif overall_min_error > thresholds['medium']:
            self._add_unique_tag(sample, "MPE_fov>25")
        elif overall_min_error > thresholds['low']:
            self._add_unique_tag(sample, "MPE_fov>10")
        elif overall_min_error > thresholds['very_low']:
            self._add_unique_tag(sample, "MPE_fov>5")
        else:
            self._add_unique_tag(sample, "MPE_fov<5")
    
    def _add_unique_tag(self, sample: fo.Sample, tag: str):
        """Add tag to sample if not already present."""
        if tag not in sample.tags:
            sample.tags.append(tag) 