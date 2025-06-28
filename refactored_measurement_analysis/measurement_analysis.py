#!/usr/bin/env python3
"""Core measurement analysis orchestration."""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import math
from tqdm import tqdm

from .config import Config
from .models import ObjectLengthMeasurer, FocalLengthMeasurer, PrawnMeasurements, MeasurementResult
from .data_processing import (
    PrawnDataProcessor, YOLOParser, FilePathResolver, 
    FilenameProcessor, BoundingBoxProcessor
)
from .visualization import (
    FiftyOneDatasetManager, FiftyOneSampleProcessor, 
    VisualizationCreator, ErrorTagManager
)


class MeasurementAnalyzer:
    """
    Main class orchestrating the complete measurement analysis workflow.
    
    This class replaces the monolithic functions in measurements_analysis.py
    and data_loader.py, providing a clean interface for the entire analysis process.
    
    Key improvements from original implementation:
    - Separated data loading, processing, measurement calculation, and visualization
    - Unified carapace and body measurement workflows
    - Better error handling and progress reporting
    - Modular design allowing easy extension and testing
    """
    
    def __init__(self, measurement_type: str, weights_type: str, 
                 port: Optional[int] = None, verbose: bool = False):
        """
        Initialize the measurement analyzer.
        
        Args:
            measurement_type: Type of measurement ('carapace' or 'body')
            weights_type: Type of weights ('car', 'kalkar', 'all')
            port: Optional port for FiftyOne app
            verbose: Enable verbose logging
        """
        self.measurement_type = measurement_type
        self.weights_type = weights_type
        self.verbose = verbose
        
        # Initialize configuration and components
        self.config = Config()
        self.data_processor = PrawnDataProcessor(measurement_type)
        self.file_resolver = FilePathResolver(self.config)
        
        # Initialize measurement calculator
        self.measurer = ObjectLengthMeasurer(
            image_width=self.config.CAMERA.image_width,
            image_height=self.config.CAMERA.image_height,
            horizontal_fov=self.config.CAMERA.horizontal_fov,
            vertical_fov=self.config.CAMERA.vertical_fov,
            distance_mm=1000  # Default distance, will be updated per measurement
        )
        
        # Initialize FiftyOne components
        dataset_name = self.config.get_dataset_name(measurement_type, weights_type)
        self.dataset_manager = FiftyOneDatasetManager(dataset_name, port)
        self.sample_processor = FiftyOneSampleProcessor(self.config)
        self.visualization_creator = VisualizationCreator(self.config)
        self.error_tag_manager = ErrorTagManager(self.config)
        
        # Results storage
        self.results_data = []
        self.processed_measurements = []
    
    def _find_closest_detection_match(self, measurements: List[PrawnMeasurements]) -> List[PrawnMeasurements]:
        """
        Find closest detection matches between predictions and ground truth.
        
        This method implements the matching logic that was embedded in the original
        analysis functions, now extracted for better maintainability.
        
        Args:
            measurements: List of prawn measurements
            
        Returns:
            Updated measurements with matched detections
        """
        if self.verbose:
            print("Finding closest detection matches...")
        
        measurement_config = self.config.MEASUREMENT_CONFIGS[self.measurement_type]
        target_keypoints = [0, 1] if self.measurement_type == 'carapace' else [2, 3]
        
        for measurement in tqdm(measurements, desc="Processing detections", disable=not self.verbose):
            # Find prediction file
            pred_file = self.file_resolver.find_prediction_file(
                measurement.filename, self.weights_type
            )
            
            # Find ground truth file
            gt_file = self.file_resolver.find_ground_truth_file(measurement.filename)
            
            # Parse prediction detections
            pred_detections = []
            if pred_file:
                pred_detections = YOLOParser.parse_file(pred_file)
            
            # Parse ground truth detections
            gt_detections = []
            if gt_file:
                gt_detections = YOLOParser.parse_file(gt_file)
            
            # Find best detections
            best_pred = YOLOParser.find_best_detection(pred_detections, target_keypoints)
            best_gt = YOLOParser.find_best_detection(gt_detections, target_keypoints)
            
            # Calculate measurements from detections
            if best_pred:
                measurement.predicted_measurement = self._calculate_measurement_from_detection(
                    best_pred, measurement, target_keypoints
                )
            
            if best_gt:
                measurement.ground_truth_measurement = self._calculate_measurement_from_detection(
                    best_gt, measurement, target_keypoints
                )
            
            # Calculate focal length measurement for comparison
            if best_pred and measurement.manual_scales:
                measurement.focal_length_measurement = self._calculate_focal_length_measurement(
                    best_pred, measurement, target_keypoints
                )
        
        return measurements
    
    def _calculate_measurement_from_detection(self, detection, measurement: PrawnMeasurements, 
                                           target_keypoints: List[int]) -> Optional[MeasurementResult]:
        """
        Calculate measurement from pose detection.
        
        Args:
            detection: Pose detection object
            measurement: PrawnMeasurements object
            target_keypoints: List of keypoint indices to use
            
        Returns:
            MeasurementResult or None if calculation fails
        """
        try:
            # Get keypoints for measurement
            if len(target_keypoints) != 2 or any(idx >= len(detection.keypoints) for idx in target_keypoints):
                return None
            
            kp1 = detection.keypoints[target_keypoints[0]]
            kp2 = detection.keypoints[target_keypoints[1]]
            
            # Check keypoint confidence
            if kp1[2] < 0.5 or kp2[2] < 0.5:
                return None
            
            # Convert normalized coordinates to pixel coordinates
            point1 = (
                kp1[0] * self.config.CAMERA.image_width,
                kp1[1] * self.config.CAMERA.image_height
            )
            point2 = (
                kp2[0] * self.config.CAMERA.image_width,
                kp2[1] * self.config.CAMERA.image_height
            )
            
            # Use appropriate distance based on pond type
            if measurement.pond_type in ['left', 'right']:
                self.measurer.distance_mm = 1000  # 1000mm for left/right ponds
                self.measurer.scale_x, self.measurer.scale_y = self.measurer._calculate_scaling_factors()
            else:
                self.measurer.distance_mm = 1200  # 1200mm for car pond
                self.measurer.scale_x, self.measurer.scale_y = self.measurer._calculate_scaling_factors()
            
            # Calculate measurement
            return self.measurer.compute_length_between_points(point1, point2)
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating measurement for {measurement.filename}: {e}")
            return None
    
    def _calculate_focal_length_measurement(self, detection, measurement: PrawnMeasurements,
                                          target_keypoints: List[int]) -> Optional[float]:
        """
        Calculate measurement using focal length method for comparison.
        
        Args:
            detection: Pose detection object
            measurement: PrawnMeasurements object
            target_keypoints: List of keypoint indices to use
            
        Returns:
            Focal length measurement in mm or None if calculation fails
        """
        try:
            # Get keypoints
            kp1 = detection.keypoints[target_keypoints[0]]
            kp2 = detection.keypoints[target_keypoints[1]]
            
            # Calculate pixel distance
            delta_x = (kp2[0] - kp1[0]) * self.config.CAMERA.image_width
            delta_y = (kp2[1] - kp1[1]) * self.config.CAMERA.image_height
            pixel_distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
            
            # Use appropriate focal length and distance
            if measurement.pond_type in ['left', 'right']:
                focal_length = self.config.CAMERA.focal_length_left_right
                distance_mm = 1000
            else:
                focal_length = self.config.CAMERA.focal_length_default
                distance_mm = 1200
            
            return FocalLengthMeasurer.calculate_real_width(
                focal_length=focal_length,
                height_mm=distance_mm,
                pixel_distance=pixel_distance,
                pixel_size=self.config.CAMERA.pixel_size
            )
            
        except Exception as e:
            if self.verbose:
                print(f"Error calculating focal length measurement for {measurement.filename}: {e}")
            return None
    
    def _create_results_dataframe(self, measurements: List[PrawnMeasurements]) -> pd.DataFrame:
        """
        Create comprehensive results dataframe from measurements.
        
        This method consolidates the dataframe creation logic that was scattered
        across the original analysis functions.
        
        Args:
            measurements: List of processed measurements
            
        Returns:
            DataFrame with comprehensive analysis results
        """
        results_data = []
        
        for measurement in measurements:
            if not measurement.predicted_measurement:
                continue
            
            # Basic data
            row_data = {
                'prawn_id': measurement.prawn_id,
                'filename': measurement.filename,
                'pond_type': measurement.pond_type,
                'manual_min_length': measurement.min_length,
                'manual_max_length': measurement.max_length,
                'manual_median_length': measurement.median_length,
                'predicted_length_mm': measurement.predicted_measurement.distance_mm,
                'predicted_length_px': measurement.predicted_measurement.distance_pixels,
                'measurement_angle': measurement.predicted_measurement.angle_deg,
                'combined_scale': measurement.predicted_measurement.combined_scale,
            }
            
            # Ground truth data
            if measurement.ground_truth_measurement:
                row_data.update({
                    'ground_truth_length_mm': measurement.ground_truth_measurement.distance_mm,
                    'ground_truth_length_px': measurement.ground_truth_measurement.distance_pixels,
                    'ground_truth_angle': measurement.ground_truth_measurement.angle_deg,
                })
            
            # Focal length data
            if measurement.focal_length_measurement:
                row_data['focal_length_measurement'] = measurement.focal_length_measurement
            
            # Error calculations
            errors = measurement.calculate_errors()
            row_data.update(errors)
            
            # Manual measurements
            for i, length in enumerate(measurement.manual_lengths[:3], 1):
                row_data[f'Length_{i}'] = length
            
            for i, scale in enumerate(measurement.manual_scales[:3], 1):
                row_data[f'Scale_{i}'] = scale
            
            results_data.append(row_data)
        
        return pd.DataFrame(results_data)
    
    def _create_fiftyone_dataset(self, measurements: List[PrawnMeasurements]) -> None:
        """
        Create FiftyOne dataset with visual annotations.
        
        This method consolidates the FiftyOne dataset creation that was mixed
        with analysis logic in the original code.
        
        Args:
            measurements: List of processed measurements
        """
        if self.verbose:
            print("Creating FiftyOne dataset...")
        
        # Create or load dataset
        dataset = self.dataset_manager.create_or_load_dataset(overwrite=True)
        
        # Process measurements into samples
        samples = []
        for measurement in tqdm(measurements, desc="Creating samples", disable=not self.verbose):
            if not measurement.predicted_measurement:
                continue
            
            # Create sample
            sample = self.sample_processor.create_sample_from_measurement(measurement)
            if not sample:
                continue
            
            # Add error-based tags
            min_error = measurement.get_min_error_percentage()
            self.error_tag_manager.tag_sample_by_error(sample, min_error)
            
            # Find detection for visualization
            pred_file = self.file_resolver.find_prediction_file(
                measurement.filename, self.weights_type
            )
            
            if pred_file:
                detections = YOLOParser.parse_file(pred_file)
                if detections:
                    # Add pose detections
                    self.sample_processor.add_pose_detections(
                        sample, detections, "predictions"
                    )
                    
                    # Add measurement polylines
                    best_detection = detections[0]  # Use first detection for visualization
                    polylines = self.visualization_creator.create_measurement_polylines(
                        measurement, best_detection
                    )
                    
                    if polylines:
                        sample["measurement_lines"] = polylines[0]
                    
                    # Add skeleton polylines
                    skeleton_lines = self.visualization_creator.create_skeleton_polylines(
                        best_detection, self.measurement_type
                    )
                    
                    if skeleton_lines:
                        sample["skeleton"] = skeleton_lines
            
            samples.append(sample)
        
        # Add samples to dataset
        if samples:
            dataset.add_samples(samples)
            self.dataset_manager.persist_dataset()
            
            if self.verbose:
                print(f"Created dataset with {len(samples)} samples")
        else:
            print("No valid samples created")
    
    def run_analysis(self, output_dir: Optional[Path] = None, 
                    create_fiftyone: bool = True) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Run the complete measurement analysis workflow.
        
        This is the main entry point that orchestrates the entire analysis process,
        replacing the monolithic functions in the original code.
        
        Args:
            output_dir: Optional output directory for results
            create_fiftyone: Whether to create FiftyOne dataset
            
        Returns:
            Tuple of (results_dataframe, dataset_name)
        """
        if self.verbose:
            print(f"Starting {self.measurement_type} measurement analysis with {self.weights_type} weights")
        
        # Step 1: Load and process measurement data
        if self.verbose:
            print("Loading measurement data...")
        measurements = self.data_processor.process_all_measurements()
        print(f"Loaded {len(measurements)} measurements")
        
        # Step 2: Find closest detection matches
        measurements = self._find_closest_detection_match(measurements)
        
        # Filter measurements with predictions
        valid_measurements = [m for m in measurements if m.predicted_measurement]
        print(f"Found predictions for {len(valid_measurements)} measurements")
        
        if not valid_measurements:
            print("No valid measurements found. Exiting.")
            return pd.DataFrame(), None
        
        # Step 3: Create results dataframe
        if self.verbose:
            print("Creating results dataframe...")
        results_df = self._create_results_dataframe(valid_measurements)
        
        # Step 4: Save results to file
        if output_dir:
            output_file = output_dir / self.config.get_output_filename(
                self.measurement_type, self.weights_type
            )
            results_df.to_excel(output_file, index=False)
            if self.verbose:
                print(f"Results saved to {output_file}")
        
        # Step 5: Create FiftyOne dataset
        dataset_name = None
        if create_fiftyone:
            self._create_fiftyone_dataset(valid_measurements)
            dataset_name = self.dataset_manager.dataset_name
        
        # Store results for potential later use
        self.results_data = results_df
        self.processed_measurements = valid_measurements
        
        if self.verbose:
            print("Analysis completed successfully!")
        
        return results_df, dataset_name
    
    def launch_fiftyone_app(self, **kwargs):
        """
        Launch FiftyOne app for dataset visualization.
        
        Args:
            **kwargs: Additional arguments for FiftyOne app
        """
        if not self.dataset_manager.dataset:
            print("No dataset available. Run analysis first.")
            return None
        
        return self.dataset_manager.launch_app(**kwargs)
    
    def get_analysis_summary(self) -> Dict:
        """
        Get summary statistics of the analysis results.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.results_data.empty:
            return {}
        
        summary = {
            'total_measurements': len(self.results_data),
            'measurement_type': self.measurement_type,
            'weights_type': self.weights_type,
        }
        
        # Error statistics
        if 'error_mpe_median' in self.results_data.columns:
            error_col = self.results_data['error_mpe_median'].dropna()
            if not error_col.empty:
                summary.update({
                    'mean_error_percentage': error_col.mean(),
                    'median_error_percentage': error_col.median(),
                    'std_error_percentage': error_col.std(),
                    'min_error_percentage': error_col.min(),
                    'max_error_percentage': error_col.max(),
                })
        
        # Pond type distribution
        if 'pond_type' in self.results_data.columns:
            pond_counts = self.results_data['pond_type'].value_counts().to_dict()
            summary['pond_type_distribution'] = pond_counts
        
        return summary 