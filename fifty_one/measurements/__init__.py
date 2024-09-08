# prawn_analysis/__init__.py

import logging

# Configure logging for the package
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optionally, import key modules for easier access
from data_loader import load_data, create_dataset, process_images
from utils import parse_pose_estimation, calculate_euclidean_distance, calculate_real_width, extract_identifier_from_gt
from metrics import compute_pose_metrics, compute_dataset_map