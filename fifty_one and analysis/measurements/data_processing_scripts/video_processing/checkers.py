"""
Quality Checkers Module - Advanced Frame Quality Detection Algorithms

This module implements sophisticated quality checking algorithms for video frame filtering.
Each checker focuses on a specific aspect of frame quality to ensure only high-quality,
useful frames are selected for dataset creation.

Quality Detection Methods:
1. ThresholdBlurChecker - Laplacian variance based blur detection
2. SimilarityChecker - Optical flow based duplicate frame detection  
3. BlackFrameChecker - Luminance analysis based black frame detection
4. NaiveBlackFrameChecker - Simple average-based black frame detection

The checkers work in sequence to filter out low-quality frames, creating
clean datasets for computer vision applications.

Author: Research Team
Purpose: Underwater prawn counting and measurement research

This module includes code derived from OpenDroneMap (https://github.com/OpenDroneMap/ODM),
which is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
As such, this module is also licensed under AGPL-3.0.
"""

import cv2
import numpy as np
import os

class ThresholdBlurChecker:
    """
    Laplacian Variance Blur Detection System
    
    This class implements blur detection using the Laplacian operator variance method.
    The Laplacian operator highlights regions of rapid intensity change (edges).
    Blurry images have fewer sharp edges, resulting in lower Laplacian variance.
    
    Algorithm Details:
    1. Apply Laplacian filter to grayscale image
    2. Calculate variance of the filtered result
    3. Compare variance against threshold
    4. Low variance indicates blur (fewer sharp edges)
    
    Mathematical Foundation:
    - Laplacian kernel: [[0,-1,0],[-1,4,-1],[0,-1,0]]
    - Variance = E[(X - μ)²] where X is pixel intensity
    - Threshold typically 100-300 for good separation
    
    Performance Characteristics:
    - Fast: Single-pass algorithm
    - Accurate: Effective for most blur types
    - Robust: Works well with various image content
    """
    
    def __init__(self, threshold):
        """
        Initialize blur checker with variance threshold.
        
        Args:
            threshold (float): Laplacian variance threshold below which frames are considered blurry
                              Typical values: 100 (strict) to 300 (lenient)
                              Lower values = more sensitive to blur
        """
        self.threshold = threshold
        print(f"ThresholdBlurChecker initialized with threshold: {threshold}")

    def NeedPreProcess(self):
        """
        Indicate whether this checker requires video preprocessing.
        
        Returns:
            bool: False - blur detection works frame-by-frame without preprocessing
        """
        return False

    def PreProcess(self, video_path, start_frame, end_frame):
        """
        Preprocessing method (not required for blur detection).
        
        Args:
            video_path (str): Path to video file
            start_frame (int): Starting frame index
            end_frame (int): Ending frame index
        """
        return  # No preprocessing needed

    def IsBlur(self, image_bw, id, frame):
        """
        Detect if frame is blurry using Laplacian variance method.
        
        This method applies the Laplacian operator to detect edges and calculates
        the variance of the result. Blurry images have fewer sharp edges and thus
        lower variance in the Laplacian response.
        
        Algorithm Steps:
        1. Apply Laplacian filter: cv2.Laplacian(image, cv2.CV_64F)
        2. Calculate variance: var() of filtered result
        3. Compare against threshold: variance < threshold = blurry
        4. Optionally save blurry frames for analysis
        
        Args:
            image_bw (numpy.ndarray): Grayscale image for blur analysis
            id (int): Frame identifier for logging/debugging
            frame (numpy.ndarray): Original color frame (for optional saving)
            
        Returns:
            tuple: (variance_score, is_blurry)
                variance_score (float): Laplacian variance value (higher = sharper)
                is_blurry (bool): True if frame is below blur threshold
        """
        # Create directory for saving blurry frames (debugging/analysis)
        if not os.path.exists('blurry_images'):
            os.mkdir('blurry_images')
            
        # Apply Laplacian operator for edge detection
        # CV_64F ensures floating-point precision for variance calculation
        laplacian = cv2.Laplacian(image_bw, cv2.CV_64F)
        
        # Calculate variance of Laplacian response
        # High variance = many edges = sharp image
        # Low variance = few edges = blurry image
        var = laplacian.var()
        
        # Determine if frame is blurry based on threshold
        is_blur = var < self.threshold
        
        # Optional: Save blurry frames for analysis and threshold tuning
        # Uncomment the following lines to save detected blurry frames
        # if is_blur:
        #     print(f"Image {id} is blurry with variance {var}")
        #     cv2.imwrite(f'blurry_images/blur_{id}_{var:.2f}.jpg', frame)
            
        return var, is_blur


class SimilarityChecker:
    """
    Optical Flow Based Similarity Detection System
    
    This class detects frames that are too similar to previously processed frames
    using optical flow analysis. It tracks feature points between consecutive frames
    and measures their movement to determine similarity.
    
    Algorithm Details:
    1. Detect good features to track using Shi-Tomasi corner detection
    2. Track features between frames using Lucas-Kanade optical flow
    3. Calculate average movement distance of successfully tracked features
    4. Compare distance against threshold for similarity decision
    
    Technical Implementation:
    - Feature Detection: cv2.goodFeaturesToTrack() - Shi-Tomasi corner detector
    - Optical Flow: cv2.calcOpticalFlowPyrLK() - Lucas-Kanade method
    - Distance Metric: Average Euclidean distance of feature movement
    
    Performance Characteristics:
    - Accurate: Robust to lighting changes and minor variations
    - Efficient: Tracks limited number of features for speed
    - Adaptive: Updates reference frame when significant change detected
    """
    
    def __init__(self, threshold, max_features=500):
        """
        Initialize similarity checker with distance threshold and feature limit.
        
        Args:
            threshold (float): Average feature movement distance below which frames are similar
                              Typical values: 20 (strict) to 50 (lenient)
                              Lower values = more sensitive to similarity
            max_features (int): Maximum number of features to track for efficiency
                               Default 500 provides good balance of accuracy and speed
        """
        self.threshold = threshold
        self.max_features = max_features
        
        # State tracking for comparison
        self.last_image = None         # Previous frame for comparison
        self.last_image_id = None      # ID of last accepted frame
        self.last_image_features = None # Features from last accepted frame
        
        print(f"SimilarityChecker initialized with threshold: {threshold}, max_features: {max_features}")

    def IsSimilar(self, image_bw, id):
        """
        Determine if current frame is too similar to the last accepted frame.
        
        This method uses optical flow to track feature points between the current
        frame and the last accepted frame. If features haven't moved significantly,
        the frames are considered similar and the current frame is rejected.
        
        Algorithm Flow:
        1. First frame: Always accept and use as reference
        2. Subsequent frames: Track features from reference to current
        3. Calculate average movement distance of tracked features
        4. Compare distance to threshold for similarity decision
        5. Update reference frame if significantly different
        
        Args:
            image_bw (numpy.ndarray): Current grayscale frame for analysis
            id (int): Frame identifier for logging
            
        Returns:
            tuple: (distance, is_similar, last_frame_id)
                distance (float): Average feature movement distance
                is_similar (bool): True if frame is too similar to last
                last_frame_id (int): ID of reference frame used for comparison
        """
        # First frame: establish reference
        if self.last_image is None:
            self.last_image = image_bw.copy()
            self.last_image_id = id
            
            # Detect good features to track using Shi-Tomasi corner detection
            # Parameters: maxCorners, qualityLevel, minDistance
            self.last_image_features = cv2.goodFeaturesToTrack(
                image_bw, 
                self.max_features,  # Maximum number of features
                0.01,               # Quality level (0.01 = top 1% of corners)
                10                  # Minimum distance between features
            )
            
            return 0, False, None  # First frame is never similar

        # Track features from reference frame to current frame using Lucas-Kanade optical flow
        # This finds where each feature point has moved between frames
        features, status, error = cv2.calcOpticalFlowPyrLK(
            self.last_image,           # Previous frame
            image_bw,                  # Current frame  
            self.last_image_features,  # Features to track
            None                       # Output array (auto-allocated)
        )

        # Filter out features that couldn't be tracked successfully
        # status=1 indicates successful tracking, status=0 indicates failure
        good_features = features[status == 1]        # Current positions of good features
        good_features2 = self.last_image_features[status == 1]  # Previous positions

        # Calculate movement distance for each successfully tracked feature
        # Use L1 distance (Manhattan distance) for efficiency
        if len(good_features) > 0:
            distances = np.abs(good_features2 - good_features)
            distance = np.average(distances)  # Average movement distance
        else:
            distance = float('inf')  # No features tracked = very different

        # Determine similarity based on average movement distance
        is_similar = distance < self.threshold

        # Update reference frame if current frame is significantly different
        if not is_similar:
            self.last_image = image_bw.copy()
            self.last_image_id = id
            # Detect new features in the current frame for future comparisons
            self.last_image_features = cv2.goodFeaturesToTrack(
                image_bw, 
                self.max_features, 
                0.01, 
                10
            )
        
        # Optional debugging output
        # if is_similar:
        #     print(f"Image {id} is similar to image {self.last_image_id} with distance {distance}")
        # else:
        #     print(f"Image {id} is not similar to image {self.last_image_id} with distance {distance}")
            
        return distance, is_similar, self.last_image_id


class NaiveBlackFrameChecker:
    """
    Simple Average-Based Black Frame Detection System
    
    This class provides a basic black frame detection method using simple
    average luminance calculation. It's faster than the advanced BlackFrameChecker
    but less accurate for complex scenarios.
    
    Algorithm:
    1. Calculate average pixel value across entire frame
    2. Compare average against threshold
    3. Frame is black if average is below threshold
    
    Use Cases:
    - Quick processing when speed is priority
    - Simple videos with clear black/non-black distinction
    - Initial filtering before more sophisticated analysis
    """
    
    def __init__(self, threshold):
        """
        Initialize naive black frame checker.
        
        Args:
            threshold (float): Average pixel value below which frame is considered black
                              Range 0-255, typical values: 10-30
        """
        self.threshold = threshold
        print(f"NaiveBlackFrameChecker initialized with threshold: {threshold}")

    def PreProcess(self, video_path, start_frame, end_frame, width=800, height=600):
        """
        Preprocessing method (not required for naive detection).
        
        Args:
            video_path (str): Path to video file
            start_frame (int): Starting frame index
            end_frame (int): Ending frame index
            width (int): Processing width (unused)
            height (int): Processing height (unused)
        """
        return  # No preprocessing needed

    def NeedPreProcess(self):
        """
        Indicate preprocessing requirement.
        
        Returns:
            bool: False - naive method doesn't need preprocessing
        """
        return False

    def IsBlack(self, image_bw, id):
        """
        Simple black frame detection using average pixel value.
        
        Args:
            image_bw (numpy.ndarray): Grayscale image for analysis
            id (int): Frame identifier
            
        Returns:
            bool: True if frame average is below threshold (black frame)
        """
        avg_luminance = np.average(image_bw)
        return avg_luminance < self.threshold


class BlackFrameChecker:
    """
    Advanced Luminance-Based Black Frame Detection System
    
    This class implements sophisticated black frame detection using adaptive
    luminance thresholds calculated from the entire video. It performs better
    than simple average-based methods by accounting for the video's luminance
    characteristics.
    
    Algorithm Overview:
    1. PREPROCESSING: Analyze entire video to calculate luminance statistics
    2. THRESHOLD CALCULATION: Determine adaptive black pixel threshold
    3. FRAME ANALYSIS: Count black pixels and calculate ratio
    4. DECISION: Frame is black if black pixel ratio exceeds threshold
    
    Advantages:
    - Adaptive to video characteristics
    - Robust to varying lighting conditions
    - Accounts for video-specific luminance ranges
    - More accurate than simple average methods
    
    Computational Cost:
    - High: Requires full video preprocessing pass
    - Memory efficient: Processes frames sequentially
    - One-time cost: Preprocessing done once per video
    """
    
    def __init__(self, picture_black_ratio_th=0.98, pixel_black_th=0.30):
        """
        Initialize advanced black frame checker with adaptive thresholds.
        
        Args:
            picture_black_ratio_th (float): Ratio of black pixels required for black frame
                                          Default 0.98 = 98% of pixels must be black
                                          Range: 0.0-1.0, higher = stricter
            pixel_black_th (float): Relative threshold for individual pixel blackness
                                   Default 0.30 = 30% of luminance range
                                   Range: 0.0-1.0, lower = more sensitive
        """
        self.picture_black_ratio_th = picture_black_ratio_th if picture_black_ratio_th is not None else 0.98
        self.pixel_black_th = pixel_black_th if pixel_black_th is not None else 0.30
        
        # Luminance statistics calculated during preprocessing
        self.luminance_minimum_value = None    # Darkest pixel value in video
        self.luminance_range_size = None       # Range of luminance values
        self.absolute_threshold = None         # Calculated black pixel threshold
        
        print(f"BlackFrameChecker initialized:")
        print(f"  Picture black ratio threshold: {self.picture_black_ratio_th}")
        print(f"  Pixel black threshold: {self.pixel_black_th}")

    def NeedPreProcess(self):
        """
        Indicate that this checker requires video preprocessing.
        
        Returns:
            bool: True - advanced detection requires luminance analysis
        """
        return True

    def PreProcess(self, video_path, start_frame, end_frame):
        """
        Analyze entire video to calculate adaptive luminance thresholds.
        
        This preprocessing step analyzes the entire video to determine:
        1. Minimum luminance value across all frames
        2. Maximum luminance range in any single frame
        3. Adaptive threshold for black pixel detection
        
        The preprocessing enables more accurate black frame detection by
        adapting to the specific luminance characteristics of each video.
        
        Args:
            video_path (str): Path to video file for analysis
            start_frame (int): Starting frame index for analysis
            end_frame (int): Ending frame index (None for entire video)
            
        Process:
        1. Open video and set frame range
        2. Process each frame to find min/max luminance values
        3. Calculate luminance range size across all frames
        4. Determine absolute threshold for black pixel detection
        """
        print("Starting black frame preprocessing...")
        
        # Open video file for analysis
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Set frame start and end indices
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_end = end_frame
        if end_frame == -1 or end_frame is None:
            frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize luminance statistics
        self.luminance_range_size = 0      # Maximum range found in any frame
        self.luminance_minimum_value = 255  # Minimum value found across all frames

        frame_index = start_frame if start_frame is not None else 0
        total_frames = frame_end - frame_index
        processed_frames = 0

        print(f"Analyzing {total_frames} frames for luminance statistics...")

        # Analyze each frame for luminance characteristics
        while (cap.isOpened() and (end_frame is None or frame_index <= end_frame)):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale for luminance analysis
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate frame-specific luminance statistics
            gray_frame_min = gray_frame.min()  # Darkest pixel in this frame
            gray_frame_max = gray_frame.max()  # Brightest pixel in this frame

            # Update global statistics
            # Track the maximum luminance range found in any single frame
            frame_range = gray_frame_max - gray_frame_min
            self.luminance_range_size = max(self.luminance_range_size, frame_range)
            
            # Track the minimum luminance value found across all frames
            self.luminance_minimum_value = min(self.luminance_minimum_value, gray_frame_min)

            frame_index += 1
            processed_frames += 1
            
            # Progress reporting
            if processed_frames % 100 == 0:
                progress = (processed_frames / total_frames) * 100
                print(f"Preprocessing progress: {progress:.1f}% ({processed_frames}/{total_frames})")

        # Calculate adaptive threshold for black pixel detection
        # Formula: minimum_value + (threshold_ratio * range_size)
        # This adapts the threshold to the video's specific luminance characteristics
        self.absolute_threshold = self.luminance_minimum_value + (self.pixel_black_th * self.luminance_range_size)

        # Close video file
        cap.release()
        
        print("Black frame preprocessing completed:")
        print(f"  Luminance range size: {self.luminance_range_size}")
        print(f"  Luminance minimum value: {self.luminance_minimum_value}")
        print(f"  Calculated absolute threshold: {self.absolute_threshold}")

    def IsBlack(self, image_bw, id):
        """
        Determine if frame is black using adaptive threshold analysis.
        
        This method uses the preprocessed luminance statistics to accurately
        detect black frames by counting pixels below the adaptive threshold.
        
        Algorithm:
        1. Count pixels below adaptive threshold (calculated in preprocessing)
        2. Calculate ratio of black pixels to total pixels
        3. Compare ratio against picture black ratio threshold
        4. Frame is black if ratio exceeds threshold
        
        Args:
            image_bw (numpy.ndarray): Grayscale image for black frame analysis
            id (int): Frame identifier for logging
            
        Returns:
            bool: True if frame is determined to be black
                 False if frame contains sufficient non-black content
        """
        # Count pixels below the adaptive threshold
        # These are considered "black" pixels based on video characteristics
        black_pixels = np.sum(image_bw < self.absolute_threshold)

        # Calculate total pixels in frame
        total_pixels = image_bw.shape[0] * image_bw.shape[1]
        
        # Calculate ratio of black pixels to total pixels
        black_pixel_ratio = black_pixels / total_pixels

        # Determine if frame is black based on ratio threshold
        is_black = black_pixel_ratio >= self.picture_black_ratio_th
        
        # Optional debugging output
        # if is_black:
        #     print(f"Frame {id} is black: {black_pixel_ratio:.3f} ratio ({black_pixels}/{total_pixels})")
        
        return is_black