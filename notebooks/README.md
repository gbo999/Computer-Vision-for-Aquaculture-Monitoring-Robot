# 📚 Notebooks Directory

This directory contains all Jupyter notebooks organized by functionality and purpose. All **57 notebooks** from across the project have been consolidated here for better organization and accessibility.

## 📁 Directory Structure

### 🎯 **training/** (4 notebooks)
Model training and experimentation notebooks
- `detection_yolo_wandb.ipynb` - YOLO object detection training with Weights & Biases
- `keypoint_yolo_wandb.ipynb` - YOLO keypoint detection training with W&B
- `segment_yolo_wandb.ipynb` - YOLO segmentation training with W&B  
- `yolon-nas-pose.ipynb` - YOLO-NAS pose estimation experiments

### 📊 **analysis/** (9 notebooks)
Data analysis, statistics, and results evaluation
- `analysis-shai-exuviae.ipynb` - Exuviae analysis for Shai's research
- `carapace_stats_analysisi.ipynb` - Carapace statistics analysis
- `error_flag_analysis_body.ipynb` - Body measurement error analysis
- `error_flag_analysis_carapace.ipynb` - Carapace measurement error analysis
- `exuviae_statistics_analysis.ipynb` - Statistical analysis of exuviae data
- `full_body_stats_analysis.ipynb` - Complete body statistics analysis
- `molt_length_analysis.ipynb` - Molt length statistical analysis
- `combined_files_exploration.ipynb` - Combined dataset exploration
- `chessboard_check.ipynb` - Camera calibration validation

### 📏 **measurement/** (11 notebooks)
Size and distance measurement algorithms
- `measurements.ipynb` - Core measurement calculations
- `combined.ipynb` - Combined measurement approaches
- `measurement_combined.ipynb` - Combined measurement methods
- `depth_experiment.ipynb` - Depth estimation experiments
- `molt.ipynb` - Molt measurement algorithms
- `post_processin.ipynb` - Post-processing measurement corrections
- `distance.ipynb` - Distance calculation methods
- `height_stats.ipynb` - Height measurement statistics
- `measurements_calculator.ipynb` - Measurement calculation tools
- `metrics.ipynb` - Measurement accuracy metrics
- `measurement_report.ipynb` - Comprehensive measurement report

### 🔢 **counting/** (4 notebooks)
Object counting algorithms and validation
- `blob.ipynb` - Blob detection counting methods
- `compute_errors.ipynb` - Counting error computation
- `Detections_exctract.ipynb` - Detection extraction and counting
- `counting.ipynb` - Core counting algorithms

### 🔧 **preprocessing/** (8 notebooks)
Data preprocessing and image enhancement
- `extract_frame_gyroflow.ipynb` - Frame extraction from Gyroflow
- `compare_folders.ipynb` - Dataset folder comparison
- `run_video_split.ipynb` - Video splitting utilities
- `focal.ipynb` - Focal length analysis
- `telemetry_analysis.ipynb` - Camera telemetry analysis
- `gamma_all.ipynb` - Gamma correction processing
- `image_enhancement_trials.ipynb` - Image enhancement experiments
- `streching.ipynb` - Image stretching preprocessing

### 🧪 **experiments/** (12 notebooks)
Research experiments and proof-of-concepts
- `31-12.ipynb` - December 31st experiments
- `colab.ipynb` - Google Colab experiments
- `inference-keypoint.ipynb` - Keypoint inference testing
- `inference.ipynb` - General inference experiments
- `keypoint.ipynb` - Keypoint detection experiments
- `mark _prawn.ipynb` - Prawn marking experiments
- `mark.ipynb` - Object marking experiments
- `rt_detr_square (1) copy.ipynb` - RT-DETR square detection
- `rt_detr_square (1).ipynb` - RT-DETR square detection original
- `pose_model_asses.ipynb` - Pose model assessment
- `cpu_keypoint.ipynb` - CPU-based keypoint detection
- `experiments_rectangle.ipynb` - Rectangle detection experiments

### 🔨 **utilities/** (9 notebooks)
Helper tools and utility functions
- `see_files.ipynb` - File system exploration
- `convert_keypoints_label_format.ipynb` - Label format conversion
- `keypoint_prediction_results_check.ipynb` - Prediction validation
- `mobile_sam_exploration.ipynb` - Mobile SAM model exploration
- `undistortions_experiments.ipynb` - Image undistortion experiments
- `viewing_predictions.ipynb` - Prediction visualization tools
- `video_duration.ipynb` - Video duration analysis
- `carapace_traide.ipynb` - Carapace trading analysis
- `powper_point_helper.ipynb` - PowerPoint helper utilities

## 🚀 Usage Guidelines

1. **Before Running**: Ensure you have the required dependencies installed
2. **Data Paths**: Update any hardcoded paths in notebooks to reflect the new structure
3. **Environment**: Use the project's virtual environment (`(.venv)`)
4. **Documentation**: Each notebook should have clear markdown explanations

## 📋 Next Steps

1. **Update imports** in notebooks that reference moved files
2. **Add consistent headers** to all notebooks with purpose and author info
3. **Review and clean** outdated or duplicate notebooks
4. **Add requirements** for each notebook category
5. **Create workflow documentation** for common analysis pipelines

## 🔧 Maintenance

- Keep this README updated when adding new notebooks
- Follow the established naming conventions
- Place notebooks in the most appropriate category
- Consider creating subcategories if folders become too large

---
*Organized on: {{ date }}*
*Total Notebooks: 57*
*Categories: 7* 