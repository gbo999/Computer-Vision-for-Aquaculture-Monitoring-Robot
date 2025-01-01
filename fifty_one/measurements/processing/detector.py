class PrawnDetector:
    """Handles YOLO detection and keypoint processing"""
    def process_poses(self, poses, is_ground_truth=False):
        """Process YOLO pose estimations into keypoints and detections""" 