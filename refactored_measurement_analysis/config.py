#!/usr/bin/env python3
"""Configuration settings for prawn measurement analysis."""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CameraConfig:
    """Camera configuration parameters."""
    image_width: int = 5312
    image_height: int = 2988
    horizontal_fov: float = 76.2
    vertical_fov: float = 46.0
    focal_length_default: float = 24.72
    focal_length_left_right: float = 23.64
    pixel_size: float = 0.00716844  # mm


@dataclass
class MeasurementConfig:
    """Measurement type configurations."""
    keypoint_classes: List[str]
    skeleton_labels: List[str]
    skeleton_edges: List[List[int]]
    label_prefix: str


class Config:
    """Main configuration class."""
    
    # Base paths
    BASE_PATH = Path("/Users/gilbenor/Documents/code_projects/msc/counting_research_algorithms")
    ONEDRIVE_BASE = Path("/Users/gilbenor/Library/CloudStorage/OneDrive-post.bgu.ac.il")
    
    # Data paths
    MEASUREMENTS_PATH = ONEDRIVE_BASE / "measurements"
    THESIS_EXPORT_PATH = ONEDRIVE_BASE / "thesisi/thesis document"
    
    # Image directories
    IMAGE_PATHS = {
        'right': ONEDRIVE_BASE / "measurements/carapace/right",
        'left': ONEDRIVE_BASE / "measurements/carapace/left",
        'car': ONEDRIVE_BASE / "measurements/carapace/car"
    }
    
    # Data files
    CARAPACE_DATA_PATH = BASE_PATH / "data_files/measurement/Filtered_Data.csv"
    BODY_DATA_PATH = BASE_PATH / "data_files/measurement/final_full_statistics_with_prawn_ids_and_uncertainty - Copy.xlsx"
    METADATA_PATH = BASE_PATH / "data_files/measurement/test images.xlsx"
    GROUND_TRUTH_PATH = Path("/Users/gilbenor/Downloads/Giant freshwater prawn carapace keypoint detection.v91i.yolov8/all/labels")
    
    # Prediction paths by weight type
    PREDICTION_PATHS = {
        'car': BASE_PATH / "runs/pose/predict88/labels",
        'kalkar': BASE_PATH / "runs/pose/predict90/labels",
        'all': BASE_PATH / "runs/pose/predict89/labels"
    }
    
    # Camera configuration
    CAMERA = CameraConfig()
    
    # Measurement type configurations
    MEASUREMENT_CONFIGS = {
        'carapace': MeasurementConfig(
            keypoint_classes=["start-carapace", "eyes"],
            skeleton_labels=["start_carapace", "eyes", "rostrum", "tail"],
            skeleton_edges=[[0, 1], [1, 2], [0, 3]],
            label_prefix="carapace"
        ),
        'body': MeasurementConfig(
            keypoint_classes=["tail", "rostrum"],
            skeleton_labels=["start_carapace", "eyes", "rostrum", "tail"],
            skeleton_edges=[[0, 1], [1, 2], [0, 3]],
            label_prefix="full body"
        )
    }
    
    # FiftyOne settings
    DEFAULT_PORT_RANGE = (5150, 5190)
    
    # Error thresholds for tagging
    ERROR_THRESHOLDS = {
        'high': 50,
        'medium': 25,
        'low': 10,
        'very_low': 5
    }
    
    @classmethod
    def get_prediction_path(cls, weights_type: str) -> Path:
        """Get prediction path for given weights type."""
        return cls.PREDICTION_PATHS.get(weights_type, cls.PREDICTION_PATHS['all'])
    
    @classmethod
    def get_output_filename(cls, measurement_type: str, weights_type: str) -> str:
        """Generate output filename for results."""
        return f'updated_filtered_data_with_lengths_{measurement_type}-{weights_type}.xlsx'
    
    @classmethod
    def get_dataset_name(cls, measurement_type: str, weights_type: str) -> str:
        """Generate dataset name for FiftyOne."""
        return f"prawn_dataset_{measurement_type}_{weights_type}"

    @classmethod
    def create_sample_from_measurement(cls, measurement_type: str, weights_type: str, original_filename: str, cleaned_filename: str) -> dict:
        """Create a sample dictionary from a measurement."""
        image_path = cls.IMAGE_PATHS.get(measurement_type)
        if not image_path or not image_path.exists():
            print(f"Image not found for original: {original_filename} | cleaned: {cleaned_filename}")
            return None

        # 1. Try Excel-based lookup using the full filename (including 'undistorted_' if present)
        if cls.test_images_metadata is not None:
            lookup_key = original_filename.split('.')[0]  # Remove extension if present
            if 'file_name_new' in cls.test_images_metadata.columns:
                match = cls.test_images_metadata[cls.test_images_metadata['file_name_new'] == lookup_key]
                if not match.empty:
                    if 'Label' in match.columns:
                        label_val = match.iloc[0]['Label']
                        if isinstance(label_val, str):
                            if ':' in label_val:
                                image_filename = label_val.split(':')[1]
                            else:
                                image_filename = label_val
                            # Search all configured image folders for an exact match (with or without extension)
                            for folder in cls.IMAGE_PATHS.values():
                                if folder.exists():
                                    for file in folder.iterdir():
                                        if file.is_file():
                                            if file.name == image_filename or file.stem == Path(image_filename).stem:
                                                return {
                                                    'image_path': file,
                                                    'label': image_filename
                                                }
        # 2. Fallback: Try to find the file by the original filename in all image folders
        for folder in cls.IMAGE_PATHS.values():
            if folder.exists():
                for file in folder.iterdir():
                    if file.is_file():
                        if file.name == original_filename or file.stem == Path(original_filename).stem:
                            return {
                                'image_path': file,
                                'label': original_filename
                            }
        # 3. Not found
        return None

        if sample is not None:
            samples.append(sample)

        samples = [sample for sample in all_samples if sample is not None] 