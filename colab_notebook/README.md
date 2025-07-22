# 🚀 Training Notebooks

This directory contains well-documented Jupyter notebooks for training computer vision models for prawn detection and analysis.

## 📚 Available Notebooks

### `prawn_keypoint_detection_yolov8_colab.ipynb`
**Purpose:** Train YOLOv8 pose estimation model for prawn keypoint detection  
**Environment:** Google Colab with GPU acceleration  
**Features:**
- ✅ Secure API key management using Colab secrets
- ✅ Custom W&B callback for advanced logging and checkpointing
- ✅ Comprehensive documentation and error handling
- ✅ Automated dataset downloading from Roboflow
- ✅ Optimized hyperparameters for prawn detection

## 🔧 Required Files

### `YoloToWandbKeypoint.py`
Custom Weights & Biases callback script that provides:
- **Automatic checkpoint saving** - Saves best models as W&B artifacts
- **Error analysis** - Computes RMSE, MAE, MRD for counting accuracy
- **Visualization** - Creates prediction comparison plots
- **Metrics tracking** - Logs comprehensive training metrics

## 🔐 Security Setup

### Google Colab Secrets
Before running the notebooks, add these secrets in Colab:

1. Click the key icon (🔑) in the left sidebar
2. Add secrets:
   - `WANDB_API_KEY`: Your W&B API key from [wandb.ai/authorize](https://wandb.ai/authorize)
   - `ROBOFLOW_API_KEY`: Your Roboflow API key from [app.roboflow.com](https://app.roboflow.com)

### Why Use Secrets?
- ✅ **Never expose API keys** in code or version control
- ✅ **Secure storage** - Encrypted in Colab environment
- ✅ **Easy rotation** - Update keys without changing code
- ✅ **Best practice** - Industry standard for API key management

## 🚀 Quick Start

1. **Open in Google Colab**
   ```
   https://colab.research.google.com/github/gbo999/counting_research_algorithms/blob/main/notebooks/training/prawn_keypoint_detection_yolov8_colab.ipynb
   ```

2. **Setup API Keys** (see Security Setup above)

3. **Run All Cells** - The notebook will:
   - Install dependencies
   - Download the custom callback script
   - Setup W&B experiment tracking
   - Download dataset from Roboflow
   - Train YOLOv8 model with optimal settings
   - Save best checkpoints to W&B

## 📊 Expected Outputs

- **W&B Dashboard** - Complete training metrics and visualizations
- **Model Artifacts** - Best checkpoints saved as W&B artifacts
- **Error Analysis** - RMSE, MAE, MRD metrics for counting accuracy
- **Prediction Plots** - Ground truth vs prediction comparisons

## 🔄 Workflow Integration

This notebook is part of the complete prawn analysis pipeline:
1. **Data Collection** - Underwater footage and annotations
2. **Training** (this notebook) - YOLOv8 keypoint detection
3. **Inference** - Apply trained model to new images
4. **Measurement** - Extract morphometric data from keypoints
5. **Analysis** - Population studies and research insights

## 🛠️ Customization

### Hyperparameters
Modify training parameters in the training cell:
```python
results = model.train(
    data=dataset_config,
    epochs=300,        # Adjust training duration
    imgsz=640,         # Modify input image size
    batch=8,           # Change batch size based on GPU memory
    patience=50,       # Early stopping patience
    seed=42            # Random seed for reproducibility
)
```

### Dataset
Update dataset configuration:
- Change Roboflow project/version
- Modify data.yaml paths
- Adjust class names and keypoint definitions

### Model Architecture
Switch between YOLOv8 variants:
- `yolov8n-pose.pt` - Nano (fastest, lowest accuracy)
- `yolov8s-pose.pt` - Small
- `yolov8m-pose.pt` - Medium
- `yolov8l-pose.pt` - Large (default, best accuracy)
- `yolov8x-pose.pt` - Extra Large (slowest, highest accuracy)

## 📈 Performance Tips

1. **GPU Selection** - Use high-RAM GPU in Colab Pro
2. **Batch Size** - Adjust based on available GPU memory
3. **Image Size** - Balance between accuracy and speed
4. **Early Stopping** - Use patience parameter to avoid overfitting
5. **Data Augmentation** - Built into YOLOv8 training pipeline

## 🐛 Troubleshooting

### Common Issues

**ImportError: YoloToWandbKeypoint**
- Ensure the callback script downloaded correctly
- Check internet connection for GitHub download
- Verify file exists in current directory

**API Key Errors**
- Verify secrets are set correctly in Colab
- Check API key validity on respective platforms
- Ensure keys have proper permissions

**Out of Memory Errors**
- Reduce batch size
- Use smaller image size
- Switch to smaller model variant

**Dataset Download Fails**
- Check Roboflow API key and permissions
- Verify project name and version
- Ensure sufficient disk space
