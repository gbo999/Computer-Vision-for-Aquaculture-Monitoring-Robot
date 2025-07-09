"""
YOLOv8 Weights & Biases Integration Callback

This module provides a custom callback class for YOLOv8 training that integrates with
Weights & Biases (wandb) for experiment tracking, model logging, and checkpoint management.

The callback handles:
- Training run initialization and configuration logging
- Model checkpoint uploading (best/last models)
- Validation metrics logging
- Prediction visualization and error analysis
- Dataset artifact management

Author: Research Team
Date: 2025
Version: 2.0 (Cleaned and Documented)
"""

import os
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import wandb
from ultralytics import YOLO


class YOLOv8WeightsBiasesIntegrationCallback:
    """
    A custom callback class for YOLOv8 training that integrates with Weights & Biases.
    
    This callback provides comprehensive experiment tracking including:
    - Model checkpoints (best/last)
    - Training metrics and validation results
    - Prediction visualizations and error analysis
    - Dataset artifact logging
    
    Attributes:
        yolo (YOLO): The YOLOv8 model instance
        run_name (str, optional): Custom name for the wandb run
        project (str, optional): W&B project name
        tags (List[str], optional): Tags for the run
        resume (str, optional): Resume strategy for existing runs
        run: The active wandb run instance
    
    Example:
        ```python
        from ultralytics import YOLO
        from yolo_wandb_callback import YOLOv8WandbCallback
        
        model = YOLO("your_model_path.pt")  # Placeholder for model path
        wandb_callback = YOLOv8WandbCallback(
            model,
            project="your_project_name",  # Placeholder for project name
            tags=["your_tag1", "your_tag2"]  # Placeholder for tags
        )
        
        # Add callbacks to model
        for event, callback_fn in wandb_callback.callbacks.items():
            model.add_callback(event, callback_fn)
        
        # Train with wandb logging
        model.train(data="data.yaml", epochs=100)
        ```
    """
    
    def __init__(
        self,
        yolo: YOLO,
        run_name: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        resume: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the YOLOv8 W&B callback.
        
        Args:
            yolo: YOLOv8 model instance
            run_name: Custom name for the W&B run (defaults to trainer.args.name)
            project: W&B project name (defaults to "YOLOv8")
            tags: List of tags for the run (defaults to ["YOLOv8"])
            resume: Resume strategy - "allow", "must", "never", or None
            **kwargs: Additional arguments passed to wandb.init()
        """
        self.yolo = yolo
        self.run_name = run_name
        self.project = project
        self.tags = tags or ["YOLOv8"]
        self.resume = resume
        self.kwargs = kwargs
        self.run = None
    
    def on_pretrain_routine_start(self, trainer: Any) -> None:
        """
        Initialize wandb run at the start of training.
        
        This callback is triggered before training begins and sets up the W&B run
        with proper configuration logging.
        
        Args:
            trainer (Any): YOLOv8 trainer instance containing training configuration
        """
        self.run = wandb.init(
            name=self.run_name or trainer.args.name,
            project=self.project or trainer.args.project or "YOLOv8",
            tags=self.tags,
            config=vars(trainer.args),
            save_code=True,
            resume=self.resume,
            **self.kwargs,
        )
        
        print(f"✓ W&B run initialized: {self.run.name}")
        print(f"✓ Project: {self.run.project}")
        print(f"✓ Run URL: {self.run.url}")
    
    def on_model_save(self, trainer: Any) -> None:
        """
        Upload model checkpoint to W&B when a new best model is saved.
        
        This is the core functionality - whenever YOLOv8 saves a new best model
        (based on fitness/validation metrics), this callback uploads it as a
        W&B artifact with proper versioning.
        
        Args:
            trainer (Any): YOLOv8 trainer instance with model and training state
        """
        if trainer.best_fitness == trainer.fitness and trainer.best.exists():
            # This is a new best model - upload to W&B
            artifact_name = f"{self.run.name}_{trainer.args.task}_best.pt"
            
            print(f"📤 Uploading best model checkpoint to W&B...")
            print(f"   Model path: {trainer.best}")
            print(f"   Fitness: {trainer.fitness:.4f}")
            print(f"   Epoch: {trainer.epoch + 1}")
            
            self.run.log_artifact(
                str(trainer.best),
                type="model",
                name=artifact_name,
                aliases=["best", f"epoch_{trainer.epoch + 1}"],
                metadata={
                    "fitness": float(trainer.fitness),
                    "epoch": trainer.epoch + 1,
                    "task": trainer.args.task,
                    "model_size": os.path.getsize(trainer.best),
                }
            )
            
            print(f"✓ Best model uploaded successfully!")
    
    def on_val_end(self, validator: Any) -> None:
        """
        Log validation metrics and results to W&B.
        
        This callback processes validation results and uploads metrics
        and visualization plots to W&B for monitoring training progress.
        
        Args:
            validator (Any): YOLOv8 validator instance with validation results
        """
        # Only log validation metrics when not in training mode
        if not validator.training:
            results = validator.metrics.mean_results()
            
            # Log validation metrics
            validation_metrics = {
                "validation/precision": results[0],
                "validation/recall": results[1], 
                "validation/mAP50": results[2],
                "validation/mAP50-95": results[3],
            }
            
            # Update run summary with final validation metrics
            self.run.summary.update(validation_metrics)
            
            # Upload validation plots if available
            plot_files = list(validator.save_dir.glob("*.png"))
            if plot_files:
                self.run.log({
                    "validation_plots": [
                        wandb.Image(str(plot_path), caption=plot_path.stem)
                        for plot_path in plot_files
                    ]
                })
            
            print(f"✓ Validation metrics logged to W&B")
            print(f"   mAP50: {results[2]:.4f}")
            print(f"   mAP50-95: {results[3]:.4f}")
    
    def on_pred_end(self, predictor: Any) -> None:
        """
        Log prediction results and analysis to W&B.
        
        This callback processes prediction outputs, computes error metrics,
        and uploads prediction visualizations and analysis results.
        
        Args:
            predictor (Any): YOLOv8 predictor instance with prediction results
        """
        if predictor.save_dir and os.path.exists(predictor.save_dir):
            print(f"📊 Processing prediction results...")
            
            # Upload prediction folder as artifact
            prediction_artifact = wandb.Artifact(
                name=f"predictions_{self.run.name}",
                type="predictions",
                description="Model prediction results and visualizations"
            )
            prediction_artifact.add_dir(str(predictor.save_dir))
            self.run.log_artifact(prediction_artifact)
            
            print(f"✓ Prediction results uploaded to W&B")
    
    def on_train_start(self, trainer: Any) -> None:
        """
        Handle training start events and dataset artifact usage.
        
        Args:
            trainer (Any): YOLOv8 trainer instance
        """
        # Optionally use dataset artifacts if available
        try:
            # Attempt to use dataset artifact - modify as needed for your setup
            dataset_artifact = self.run.use_artifact('dataset:latest')
            print(f"✓ Using dataset artifact: {dataset_artifact.name}")
        except Exception as e:
            print(f"ℹ️  No dataset artifact found or accessible: {e}")
    
    @property
    def callbacks(self) -> Dict[str, Callable]:
        """
        Return dictionary of callback functions to register with YOLOv8.
        
        Returns:
            Dict[str, Callable]: Mapping callback event names to functions.
        """
        return {
            "on_pretrain_routine_start": self.on_pretrain_routine_start,
            "on_model_save": self.on_model_save,
            "on_val_end": self.on_val_end,
            "on_pred_end": self.on_pred_end,
            "on_train_start": self.on_train_start,
        }


# Utility functions for advanced prediction analysis
def read_yolo_labels(
    label_path: Union[str, Path]
) -> List[Tuple[int, float, float, float, float]]:
    """
    Read YOLO format labels from a text file.
    
    Args:
        label_path (Union[str, Path]): Path to YOLO format label file
        
    Returns:
        List[Tuple[int, float, float, float, float]]: List of tuples containing
        (class_id, x_center, y_center, width, height). All coordinates are
        normalized (0-1).
    """
    labels = []
    label_path = Path(label_path)
    
    if label_path.exists():
        with open(label_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    labels.append((class_id, x_center, y_center, width, height))
    
    return labels


def compute_detection_errors(
    ground_truth_dir: Union[str, Path],
    predictions_dir: Union[str, Path]
) -> Dict[str, float]:
    """
    Compute detection error metrics by comparing ground truth and predicted counts.
    
    Args:
        ground_truth_dir (Union[str, Path]): Directory with ground truth labels.
        predictions_dir (Union[str, Path]): Directory with predicted labels.
        
    Returns:
        Dict[str, float]: Dictionary containing RMSE, MAE, MRD, and sample count.
    """
    gt_dir = Path(ground_truth_dir)
    pred_dir = Path(predictions_dir)
    
    squared_errors = []
    absolute_errors = []
    relative_differences = []
    
    # Process all ground truth files
    for gt_file in gt_dir.glob("*.txt"):
        pred_file = pred_dir / gt_file.name
        
        # Count objects in each file
        gt_count = len(read_yolo_labels(gt_file))
        pred_count = len(read_yolo_labels(pred_file)) if pred_file.exists() else 0
        
        # Calculate errors
        error = gt_count - pred_count
        squared_errors.append(error ** 2)
        absolute_errors.append(abs(error))
        
        # Relative difference (avoid division by zero)
        if gt_count > 0:
            relative_differences.append(abs(error) / gt_count)
    
    # Compute metrics
    n_samples = len(squared_errors)
    if n_samples == 0:
        return {"RMSE": 0.0, "MAE": 0.0, "MRD": 0.0}
    
    rmse = math.sqrt(sum(squared_errors) / n_samples)
    mae = sum(absolute_errors) / n_samples
    mrd = sum(relative_differences) / len(relative_differences) if relative_differences else 0.0
    
    return {
        "RMSE": rmse,
        "MAE": mae, 
        "MRD": mrd,
        "samples": n_samples
    }


def create_detection_comparison_table(
    images_dir: Union[str, Path],
    ground_truth_dir: Union[str, Path],
    predictions_dir: Union[str, Path],
    max_images: int = 20
) -> wandb.Table:
    """
    Create a W&B table comparing ground truth and predicted detections.
    
    Args:
        images_dir (Union[str, Path]): Directory with original images.
        ground_truth_dir (Union[str, Path]): Directory with ground truth labels.
        predictions_dir (Union[str, Path]): Directory with prediction labels.
        max_images (int): Maximum number of images to include in table.
        
    Returns:
        wandb.Table: Table with comparison visualizations.
    """
    table = wandb.Table(columns=["Image", "GT_Count", "Pred_Count", "Error", "Abs_Error"])
    
    images_dir = Path(images_dir)
    gt_dir = Path(ground_truth_dir)
    pred_dir = Path(predictions_dir)
    
    processed = 0
    for img_file in images_dir.glob("*.jpg"):
        if processed >= max_images:
            break
            
        # Find corresponding label files
        label_name = img_file.stem + ".txt"
        gt_file = gt_dir / label_name
        pred_file = pred_dir / label_name
        
        # Count detections
        gt_count = len(read_yolo_labels(gt_file))
        pred_count = len(read_yolo_labels(pred_file))
        error = gt_count - pred_count
        
        # Add to table
        table.add_data(
            wandb.Image(str(img_file), caption=img_file.name),
            gt_count,
            pred_count, 
            error,
            abs(error)
        )
        
        processed += 1
    
    return table


# Backward compatibility alias
WandbCallback = YOLOv8WandbCallback


if __name__ == "__main__":
    print("YOLOv8 W&B Callback Module")
    print("This module provides W&B integration for YOLOv8 training.")
    print("Import and use the YOLOv8WandbCallback class in your training scripts.")
