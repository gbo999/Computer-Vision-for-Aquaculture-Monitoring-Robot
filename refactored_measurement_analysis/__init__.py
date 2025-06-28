"""
Refactored Prawn Measurement Analysis Package

This package provides a modular, maintainable solution for prawn measurement analysis
using computer vision and pose estimation techniques.

Key Components:
- config: Configuration management
- models: Data models and measurement calculations  
- data_processing: File I/O and data processing utilities
- visualization: FiftyOne visualization and dataset management
- measurement_analysis: Core analysis orchestration
- main: Command-line interface

Usage:
    from refactored_measurement_analysis import MeasurementAnalyzer
    
    analyzer = MeasurementAnalyzer('carapace', 'car')
    results_df, dataset_name = analyzer.run_analysis()
"""

from .measurement_analysis import MeasurementAnalyzer
from .config import Config
from .models import (
    PoseDetection, MeasurementResult, ObjectLengthMeasurer, 
    FocalLengthMeasurer, PrawnMeasurements
)

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    'MeasurementAnalyzer',
    'Config', 
    'PoseDetection',
    'MeasurementResult',
    'ObjectLengthMeasurer',
    'FocalLengthMeasurer', 
    'PrawnMeasurements'
] 