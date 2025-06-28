#!/usr/bin/env python3
"""Main measurement analysis module integrating all components."""

import socket
import random
from typing import List, Tuple, Optional
import fiftyone as fo
from pathlib import Path
from tqdm import tqdm

from config import Config
from models import ObjectLengthMeasurer, FocalLengthMeasurer, PrawnMeasurements, MeasurementResult
from data_processing import DataLoader, YOLOParser, FilePathResolver, PrawnDataProcessor, FilenameProcessor
from visualization import FiftyOneDatasetManager, FiftyOneSampleProcessor, VisualizationCreator, ErrorTagManager


class MeasurementAnalyzer:
    """
    Main measurement analyzer that coordinates all analysis components.
    
    Key improvements from original code:
    - Clean separation of concerns
    - Configurable analysis parameters
    - Better error handling
    - Extensible for different measurement types
    """
    
    def __init__(self, config: Config):
        """Initialize analyzer with configuration."""
        self.config = config
        self.data_loader = DataLoader(config)
        self.yolo_parser = YOLOParser()
        self.file_resolver = FilePathResolver(config)
        self.prawn_processor = PrawnDataProcessor(config)
        
        # FiftyOne components
        self.dataset_manager = FiftyOneDatasetManager(config)
        self.sample_processor = FiftyOneSampleProcessor(config)
        self.visualization_creator = VisualizationCreator(config)
        self.error_tag_manager = ErrorTagManager(config)
    
    def run_analysis(self, measurement_type: str, weights_type: str, port: int) -> fo.Session:
        """
        Run complete measurement analysis.
        
        Args:
            measurement_type: Type of measurement ('carapace' or 'body')
            weights_type: Type of weights to use ('car', 'kalkar', 'all')
            port: Port for FiftyOne visualization
            
        Returns:
            FiftyOne session for visualization
        """
        print(f"Starting {measurement_type} measurement analysis with {weights_type} weights")
        
        # Load data
        filtered_df, metadata_df = self.data_loader.load_measurement_data(measurement_type)
        
        # Create or load dataset
        dataset, dataset_exists = self.dataset_manager.create_or_load_dataset(measurement_type, weights_type)
        
        if dataset_exists:
            print("Dataset already exists, launching visualization...")
            return self._launch_visualization(dataset, port)
        
        # Process all test sets
        self._process_all_test_sets(
            dataset, filtered_df, metadata_df, measurement_type, weights_type
        )
        
        # Evaluate keypoint detection performance
        self._evaluate_keypoint_performance(dataset, measurement_type)
        
        # Save results
        output_filename = self.config.get_output_filename(measurement_type, weights_type)
        filtered_df.to_excel(output_filename, index=False)
        print(f"Results saved to {output_filename}")
        
        # Save and export dataset
        self.dataset_manager.save_and_export_dataset(dataset, measurement_type, weights_type)
        
        # Launch visualization
        return self._launch_visualization(dataset, port)
    
    def _process_all_test_sets(self, dataset: fo.Dataset, filtered_df, metadata_df, 
                             measurement_type: str, weights_type: str):
        """Process all test sets (right, left, car)."""
        prediction_path = self.config.get_prediction_path(weights_type)
        ground_truth_files = self.file_resolver.get_ground_truth_paths()
        
        for pond_type in ['right', 'left', 'car']:
            print(f"Processing {pond_type} pond...")
            
            # Get image paths
            image_paths = self.file_resolver.get_image_paths(pond_type)
            
            if not image_paths:
                print(f"No images found for {pond_type} pond")
                continue
            
            # Process each image
            self._process_images(
                image_paths, prediction_path, ground_truth_files,
                filtered_df, metadata_df, dataset, pond_type, measurement_type
            )
    
    def _process_images(self, image_paths: List[Path], prediction_path: Path,
                       ground_truth_files: List[Path], filtered_df, metadata_df,
                       dataset: fo.Dataset, pond_type: str, measurement_type: str):
        """Process individual images."""
        
        for image_path in tqdm(image_paths, desc=f"Processing {pond_type} images"):
            identifier = FilenameProcessor.extract_identifier(image_path.name)
            if not identifier:
                print(f"Warning: Could not extract identifier from {image_path.name}")
                continue
            
            # Find prediction and ground truth files
            prediction_file = self.file_resolver.find_prediction_file(identifier, prediction_path)
            ground_truth_file = self.file_resolver.find_ground_truth_file(identifier, ground_truth_files)
            
            if not prediction_file:
                print(f"No prediction file found for {identifier}")
                continue
            
            if not ground_truth_file:
                print(f"No ground truth found for {identifier}")
                continue
            
            # Parse YOLO files
            predictions = self.yolo_parser.parse_pose_file(prediction_file)
            ground_truths = self.yolo_parser.parse_pose_file(ground_truth_file)
            
            if not predictions or not ground_truths:
                print(f"No valid detections found for {identifier}")
                continue
            
            # Create FiftyOne sample
            sample = self.sample_processor.create_sample_with_detections(
                image_path, predictions, ground_truths, pond_type
            )
            
            # Add metadata
            self.sample_processor.add_metadata_to_sample(
                sample, identifier, metadata_df, measurement_type
            )
            
            # Process prawn measurements
            prawn_measurements = self._process_prawn_measurements(
                sample, identifier, filtered_df, measurement_type
            )
            
            # Add visualizations
            self.visualization_creator.add_measurement_visualizations(
                sample, prawn_measurements, measurement_type
            )
            
            # Add error tags
            self.error_tag_manager.add_error_tags(sample, prawn_measurements)
            
            # Add sample to dataset
            dataset.add_sample(sample)
    
    def _process_prawn_measurements(self, sample: fo.Sample, identifier: str,
                                  filtered_df, measurement_type: str) -> List[PrawnMeasurements]:
        """Process prawn measurements for a single image."""
        
        # Extract prawn measurements from dataframe
        prawn_measurements = self.prawn_processor.extract_prawn_measurements(
            filtered_df, identifier, measurement_type
        )
        
        # Calculate predicted and ground truth measurements
        for prawn_measurement in prawn_measurements:
            prawn_measurement.pond_type = sample.tags[0]
            
            # Find closest detections
            closest_pred, closest_gt = self._find_closest_detections(
                sample, prawn_measurement.bounding_boxes[0].bbox if prawn_measurement.bounding_boxes else None
            )
            
            if closest_pred and closest_gt:
                # Calculate measurements
                prawn_measurement.predicted_measurement = self._calculate_measurement(
                    closest_pred, sample, measurement_type
                )
                prawn_measurement.ground_truth_measurement = self._calculate_measurement(
                    closest_gt, sample, measurement_type
                )
                prawn_measurement.focal_length_measurement = self._calculate_focal_length_measurement(
                    closest_pred, sample, measurement_type
                )
                
                # Update detection labels
                self._update_detection_labels(closest_pred, closest_gt, prawn_measurement)
                
                # Store results in dataframe
                self._store_results_in_dataframe(
                    filtered_df, prawn_measurement, identifier, measurement_type
                )
        
        return prawn_measurements
    
    def _find_closest_detections(self, sample: fo.Sample, 
                               target_bbox: Optional[Tuple[float, float, float, float]]) -> Tuple[Optional, Optional]:
        """Find closest predicted and ground truth detections."""
        if not target_bbox:
            return None, None
        
        # Convert to normalized coordinates
        target_point = (target_bbox[0] / self.config.CAMERA.image_width, 
                       target_bbox[1] / self.config.CAMERA.image_height)
        
        # Find closest prediction
        min_distance = float('inf')
        closest_pred = None
        
        for detection in sample["detections_predictions"].detections:
            det_point = (detection.bounding_box[0], detection.bounding_box[1])
            distance = self._calculate_euclidean_distance(target_point, det_point)
            if distance < min_distance:
                min_distance = distance
                closest_pred = detection
        
        # Find closest ground truth
        min_distance = float('inf')
        closest_gt = None
        
        for detection in sample["ground_truth"].detections:
            det_point = (detection.bounding_box[0], detection.bounding_box[1])
            distance = self._calculate_euclidean_distance(target_point, det_point)
            if distance < min_distance:
                min_distance = distance
                closest_gt = detection
        
        return closest_pred, closest_gt
    
    def _calculate_measurement(self, detection, sample: fo.Sample, measurement_type: str) -> MeasurementResult:
        """Calculate measurement from detection keypoints."""
        keypoints_dict = detection.attributes["keypoints"]
        
        # Select appropriate keypoints based on measurement type
        if measurement_type == 'carapace':
            points = [keypoints_dict['start_carapace'], keypoints_dict['eyes']]
        else:  # body
            points = [keypoints_dict['tail'], keypoints_dict['rostrum']]
        
        # Scale to image dimensions
        point1_scaled = [points[0][0] * self.config.CAMERA.image_width, 
                        points[0][1] * self.config.CAMERA.image_height]
        point2_scaled = [points[1][0] * self.config.CAMERA.image_width, 
                        points[1][1] * self.config.CAMERA.image_height]
        
        # Calculate measurement
        height_mm = sample.get('height(mm)', 1000)  # Default height if not available
        measurer = ObjectLengthMeasurer(
            self.config.CAMERA.image_width,
            self.config.CAMERA.image_height,
            self.config.CAMERA.horizontal_fov,
            self.config.CAMERA.vertical_fov,
            height_mm
        )
        
        return measurer.compute_length_between_points(point1_scaled, point2_scaled)
    
    def _calculate_focal_length_measurement(self, detection, sample: fo.Sample, measurement_type: str) -> float:
        """Calculate measurement using focal length method."""
        keypoints_dict = detection.attributes["keypoints"]
        
        # Select appropriate keypoints
        if measurement_type == 'carapace':
            points = [keypoints_dict['start_carapace'], keypoints_dict['eyes']]
        else:  # body
            points = [keypoints_dict['tail'], keypoints_dict['rostrum']]
        
        # Scale to image dimensions
        point1_scaled = [points[0][0] * self.config.CAMERA.image_width, 
                        points[0][1] * self.config.CAMERA.image_height]
        point2_scaled = [points[1][0] * self.config.CAMERA.image_width, 
                        points[1][1] * self.config.CAMERA.image_height]
        
        # Calculate pixel distance
        pixel_distance = self._calculate_euclidean_distance(point1_scaled, point2_scaled)
        
        # Get focal length based on pond type
        pond_type = sample.tags[0] if sample.tags else ""
        if pond_type in ['test-left', 'test-right']:
            focal_length = self.config.CAMERA.focal_length_left_right
        else:
            focal_length = self.config.CAMERA.focal_length_default
        
        height_mm = sample.get('height(mm)', 1000)
        
        return FocalLengthMeasurer.calculate_real_width(
            focal_length, height_mm, pixel_distance, self.config.CAMERA.pixel_size
        )
    
    def _update_detection_labels(self, pred_detection, gt_detection, prawn_measurement: PrawnMeasurements):
        """Update detection labels with measurement information."""
        if prawn_measurement.predicted_measurement:
            pred_label = f'pred_length: {prawn_measurement.predicted_measurement.distance_mm:.2f}mm'
            pred_detection.label = pred_label
            pred_detection.attributes["prawn_id"] = fo.Attribute(value=prawn_measurement.prawn_id)
        
        if prawn_measurement.ground_truth_measurement:
            gt_label = f'prawn_truth: {prawn_measurement.ground_truth_measurement.distance_mm:.2f}mm'
            gt_detection.label = gt_label
    
    def _store_results_in_dataframe(self, filtered_df, prawn_measurement: PrawnMeasurements,
                                  identifier: str, measurement_type: str):
        """Store calculation results in the dataframe."""
        measurement_config = self.config.MEASUREMENT_CONFIGS[measurement_type]
        label_key = f'{measurement_config.label_prefix}:{identifier}'
        
        # Find matching row
        mask = ((filtered_df['Label'] == label_key) & 
               (filtered_df['PrawnID'] == prawn_measurement.prawn_id))
        
        if not mask.any():
            return
        
        # Store various measurements
        if prawn_measurement.predicted_measurement:
            filtered_df.loc[mask, 'Length_fov(mm)'] = prawn_measurement.predicted_measurement.distance_mm
            filtered_df.loc[mask, 'pred_Distance_pixels'] = prawn_measurement.predicted_measurement.distance_pixels
            filtered_df.loc[mask, 'combined_scale'] = prawn_measurement.predicted_measurement.combined_scale
        
        if prawn_measurement.ground_truth_measurement:
            filtered_df.loc[mask, 'Length_ground_truth_annotation(mm)'] = prawn_measurement.ground_truth_measurement.distance_mm
            filtered_df.loc[mask, 'Length_ground_truth_annotation_pixels'] = prawn_measurement.ground_truth_measurement.distance_pixels
            filtered_df.loc[mask, 'combined_scale_ground'] = prawn_measurement.ground_truth_measurement.combined_scale
        
        if prawn_measurement.focal_length_measurement:
            filtered_df.loc[mask, 'focal_RealLength(cm)'] = prawn_measurement.focal_length_measurement
        
        # Store error calculations
        errors = prawn_measurement.calculate_errors()
        for error_key, error_value in errors.items():
            column_name = self._get_error_column_name(error_key)
            filtered_df.loc[mask, column_name] = error_value
        
        # Store pond type
        filtered_df.loc[mask, 'Pond_Type'] = prawn_measurement.pond_type
    
    def _get_error_column_name(self, error_key: str) -> str:
        """Convert error key to dataframe column name."""
        error_mapping = {
            'mpe_min': 'MPError_fov_min',
            'mpe_max': 'MPError_fov_max', 
            'mpe_median': 'MPError_fov_median',
            'abs_error_min': 'AbsError_fov_min',
            'abs_error_max': 'AbsError_fov_max',
            'abs_error_median': 'AbsError_fov_median',
            'focal_mpe_min': 'MPError_focal_min',
            'focal_mpe_max': 'MPError_focal_max',
            'focal_mpe_median': 'MPError_focal_median',
            'focal_error_min': 'AbsError_focal_min',
            'focal_error_max': 'AbsError_focal_max',
            'focal_error_median': 'AbsError_focal_median',
            'gt_error_min': 'Error_distance_mm_ground_min',
            'gt_error_max': 'Error_distance_mm_ground_max',
            'gt_error_median': 'Error_distance_mm_ground_median',
            'gt_mpe_min': 'Error_percentage_distance_mm_ground_min',
            'gt_mpe_max': 'Error_percentage_distance_mm_ground_max',
            'gt_mpe_median': 'Error_percentage_distance_mm_ground_median'
        }
        return error_mapping.get(error_key, error_key)
    
    def _evaluate_keypoint_performance(self, dataset: fo.Dataset, measurement_type: str):
        """Evaluate keypoint detection performance."""
        measurement_config = self.config.MEASUREMENT_CONFIGS[measurement_type]
        
        try:
            results = dataset.evaluate_detections(
                "keypoints",
                gt_field="keypoints_truth",
                eval_key="pose_eval",
                method="coco",
                compute_mAP=True,
                iou=0.5,
                use_keypoints=True,
                classes=measurement_config.keypoint_classes
            )
            print(f"Keypoint evaluation completed for {measurement_type}")
        except Exception as e:
            print(f"Error during keypoint evaluation: {e}")
    
    def _launch_visualization(self, dataset: fo.Dataset, port: int) -> fo.Session:
        """Launch FiftyOne visualization."""
        print(f'Launching FiftyOne on port: {port}')
        session = fo.launch_app(dataset, port=port, remote=True)
        return session
    
    @staticmethod
    def _calculate_euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        import math
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


class PortManager:
    """
    Manages port allocation for FiftyOne sessions.
    
    Extracted from original port checking logic with improvements:
    - Better port availability checking
    - Configurable port ranges
    - Error handling
    """
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
    
    def get_available_port(self) -> int:
        """Get an available port in the configured range."""
        start_port, end_port = self.config.DEFAULT_PORT_RANGE
        
        # Try random port first
        port = random.randint(start_port, end_port)
        if self.is_port_available(port):
            return port
        
        # Try sequential search
        for port in range(start_port, end_port + 1):
            if self.is_port_available(port):
                return port
        
        # Fallback to a high port
        return random.randint(8000, 9000)
    
    @staticmethod
    def is_port_available(port: int) -> bool:
        """Check if a port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0 