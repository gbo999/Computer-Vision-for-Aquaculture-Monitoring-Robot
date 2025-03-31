import pandas as pd
import numpy as np
import os
import math
from pathlib import Path
# from data_loader import ObjectLengthMeasurer
from tqdm import tqdm


class ObjectLengthMeasurer:
    def __init__(self, image_width, image_height, horizontal_fov, vertical_fov, distance_mm):
        self.image_width = image_width
        self.image_height = image_height
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.distance_mm = distance_mm
        self.scale_x, self.scale_y = self.calculate_scaling_factors()
        # self.to_scale_x = image_width / 640  # Assuming low-res width is 640
        # self.to_scale_y = image_height / 360  # Assuming low-res height is 360

    def calculate_scaling_factors(self):
        """
        Calculate the scaling factors (mm per pixel) based on the camera's FOV and distance.
        """
        fov_x_rad = math.radians(self.horizontal_fov)
        fov_y_rad = math.radians(self.vertical_fov)
        scale_x = (2 * self.distance_mm * math.tan(fov_x_rad / 2)) / self.image_width
      
                # print(f'distance: {self.distance_mm}, scale_x: {scale_x}')
        scale_y = (2 * self.distance_mm * math.tan(fov_y_rad / 2)) / self.image_height
        return scale_x, scale_y

    def normalize_angle(self, angle):
        """
        Normalize the angle to [0°, 90°].
        """
        theta_norm = min(abs(angle % 180), 180 - abs(angle % 180))

        return  theta_norm

    def compute_length(self, predicted_length, angle_deg):
        """
        Compute the real-world length in millimeters using combined scaling factors.
        """
        angle_rad = math.radians(angle_deg)
        combined_scale = math.sqrt((self.scale_x * math.cos(angle_rad)) ** 2 + 
                                   (self.scale_y * math.sin(angle_rad)) ** 2)
        


        # print(f'predicted_length: {predicted_length}, combined_scale: {combined_scale}')


        length_mm = predicted_length * combined_scale
        return length_mm,combined_scale

    def compute_length_two_points(self, point1_low_res, point2_low_res):
        """
        Compute the real-world distance between two points in the low-resolution image.
        
        Parameters:
        - point1_low_res: Tuple (x1, y1) coordinates of the first point in low-res pixels.
        - point2_low_res: Tuple (x2, y2) coordinates of the second point in low-res pixels.
        
        Returns:
        - distance_mm: Real-world distance between the two points in millimeters.
        - angle_deg: Angle of the line connecting the two points relative to the horizontal axis in degrees.
        """
        # Calculate pixel distance in low-res image
        delta_x_low = point2_low_res[0] - point1_low_res[0]
        delta_y_low = point2_low_res[1] - point1_low_res[1]
        distance_px = math.sqrt(delta_x_low ** 2 + delta_y_low ** 2)

        
        
        
        # Calculate angle in degrees
        angle_rad = math.atan2(delta_y_low, delta_x_low)
        angle_deg = math.degrees(angle_rad)
        normalized_angle = self.normalize_angle(angle_deg)
        
        # Scale the pixel distance from low-res to high-res
        # distance_px_high = distance_px_low * self.to_scale_x  # Assuming uniform scaling
        
        # Compute real-world distance
        distance_mm,combined_scale = self.compute_length(distance_px, normalized_angle)
        

        return distance_mm, normalized_angle, distance_px,combined_scale


def calculate_euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def determine_size(total_length_mm):
    """
    Determine if a detection is big or small based on total length only.
    Big: 175-220mm          
    Small: 116-174mm
    """

    print(f"total_length_mm: {total_length_mm}")

    # Big: fixed range
    if 165<= total_length_mm <= 220:
        # print(f"BIG: {total_length_mm}")
        return "BIG"
    
    # Small: percentage range
    small_expected = 145
    small_min = small_expected * 0.8  # -20%
    small_max = small_expected * 1.2  # +20%
    if small_min <= total_length_mm <= small_max:
        # print(f"SMALL: {total_length_mm}")
        return "SMALL"
    
    return "REJECTED"

def analyze_good_detections():
    """Analyze detections from good images and save measurements to CSV"""
    # Read the review results
    review_csv = 'runs/pose/predict57/review_results.csv'
    review_df = pd.read_csv(review_csv)
    
    # Filter for good images only
    good_images = review_df[review_df['is_good'] == True]
    
    # Prepare data collection
    analysis_data = []
    labels_dir = Path('runs/pose/predict81/labels')
    
    # Image dimensions for calculations
    calc_width = 5312  # Original image width
    calc_height = 2988  # Original image height
    
    # Get list of label files
    label_files = list(labels_dir.glob('*.txt'))
    
    # Process each label file with progress bar
    for label_file in tqdm(label_files, desc="Processing label files"):
        image_name = label_file.stem
        
        # Initialize entry with all required fields
        entry = {
            'image_name': image_name,
            'big_total_length': '',
            'big_carapace_length': '',
            'big_eye_x': '',
            'big_eye_y': '',
            'small_total_length': '',
            'small_carapace_length': '',
            'small_eye_x': '',
            'small_eye_y': ''
        }
        
        # Determine height_mm based on image name (following same logic as visualize_filtered.py)
        is_circle2 = "GX010191" in image_name
        height_mm = 700 if is_circle2 else 410
        
        try:
            # Read and process the label file
            with open(label_file, 'r') as f:
                detections = f.readlines()
            
            
            for detection in detections:
                # Parse detection values
                values = list(map(float, detection.strip().split()))
                
                # Extract keypoints
                keypoints = []
                for i in range(5, len(values)-1, 3):
                    x = values[i]
                    y = values[i + 1]
                    keypoints.append([x, y])
                
                if len(keypoints) >= 4:
                    # Scale keypoints to original image dimensions
                    keypoints_calc = []
                    for kp in keypoints:
                        x = kp[0] * calc_width
                        y = kp[1] * calc_height
                        keypoints_calc.append([x, y])
                    
                    # Calculate total length in original image pixels (tail to rostrum - keypoints 3 to 2)
                    total_length_pixels = calculate_euclidean_distance(
                        keypoints_calc[3], keypoints_calc[2])
                    
                    # Calculate carapace length (keypoints 0 to 1)
                    carapace_length_pixels = calculate_euclidean_distance(
                        keypoints_calc[0], keypoints_calc[1])
                    
                    # Get eye coordinates (keypoint 0 instead of 2)
                    eye_x = keypoints_calc[0][0]
                    eye_y = keypoints_calc[0][1]
                    
                    # Convert to mm using original image dimensions and field of view
                    object_length_measurer = ObjectLengthMeasurer(5312, 2988,75,46 ,height_mm)
        
                    total_length_mm, angle_deg, total_length_pixels,combined_scale = object_length_measurer.compute_length_two_points(keypoints_calc[3], keypoints_calc[2])
    
                    carapace_length_mm, angle_deg_carapace, carapace_length_pixels,combined_scale_carapace = object_length_measurer.compute_length_two_points(keypoints_calc[0], keypoints_calc[1])
    
                   
                                    
                    # Determine size
                    size = determine_size(total_length_mm)
    
                    # Store measurements in the appropriate columns
                    if size == "BIG":
                        height_mm = 660 if is_circle2 else 370
                        object_length_measurer = ObjectLengthMeasurer(5312, 2988,75,46, height_mm)
                        total_length_mm, angle_deg, total_length_pixels,combined_scale = object_length_measurer.compute_length_two_points(keypoints_calc[3], keypoints_calc[2])
                        
                        carapace_length_mm, angle_deg_carapace, carapace_length_pixels,combined_scale_carapace = object_length_measurer.compute_length_two_points(keypoints_calc[0], keypoints_calc[1])
                        
                        entry['big_total_length'] = round(total_length_mm, 1)
                        entry['big_carapace_length'] = round(carapace_length_mm, 1)
                        entry['big_eye_x'] = round(eye_x, 1)
                        entry['big_eye_y'] = round(eye_y, 1)
                        entry['Big_pixels_total_length'] = round(total_length_pixels, 1)
                        entry['Big_pixels_carapace_length'] = round(carapace_length_pixels, 1)
                    elif size == "SMALL":
                        height_mm = 680 if is_circle2 else 390
                        object_length_measurer = ObjectLengthMeasurer(5312, 2988,75,46, height_mm)
                        total_length_mm, angle_deg, total_length_pixels,combined_scale = object_length_measurer.compute_length_two_points(keypoints_calc[3], keypoints_calc[2])
    
                        carapace_length_mm, angle_deg_carapace, carapace_length_pixels,combined_scale_carapace = object_length_measurer.compute_length_two_points(keypoints_calc[0], keypoints_calc[1])
    
                        entry['small_total_length'] = round(total_length_mm, 1)
                        entry['small_carapace_length'] = round(carapace_length_mm, 1)
                        entry['small_eye_x'] = round(eye_x, 1)
                        entry['small_eye_y'] = round(eye_y, 1)
                        entry['Small_pixels_total_length'] = round(total_length_pixels, 1)
                        entry['Small_pixels_carapace_length'] = round(carapace_length_pixels, 1)
            
            # Add to analysis data
            analysis_data.append(entry)
            
            # Save progress after each file
            analysis_df = pd.DataFrame(analysis_data)
            output_file = 'runs/pose/predict81/length_analysis_new.csv'
            analysis_df.to_csv(output_file, index=False)

        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue
    
    # Print summary statistics
    print(f"\nTotal files processed: {len(analysis_data)}")
    print(f"Images with big exuviae: {sum(1 for x in analysis_data if x['big_total_length'] != '')}")
    print(f"Images with small exuviae: {sum(1 for x in analysis_data if x['small_total_length'] != '')}")
    
    # Calculate statistics on lengths
    big_lengths = [entry['big_total_length'] for entry in analysis_data if entry['big_total_length'] != '']
    small_lengths = [entry['small_total_length'] for entry in analysis_data if entry['small_total_length'] != '']
    
    if big_lengths:
        print("\nBig exuviae statistics:")
        print(f"Count: {len(big_lengths)}")
        print(f"Mean total length: {np.mean(big_lengths):.1f}mm")
        print(f"Median total length: {np.median(big_lengths):.1f}mm")
        
        big_carapace = [entry['big_carapace_length'] for entry in analysis_data if entry['big_carapace_length'] != '']
        print(f"Mean carapace length: {np.mean(big_carapace):.1f}mm")
    
    if small_lengths:
        print("\nSmall exuviae statistics:")
        print(f"Count: {len(small_lengths)}")
        print(f"Mean total length: {np.mean(small_lengths):.1f}mm")
        print(f"Median total length: {np.median(small_lengths):.1f}mm")
        
        small_carapace = [entry['small_carapace_length'] for entry in analysis_data if entry['small_carapace_length'] != '']
        print(f"Mean carapace length: {np.mean(small_carapace):.1f}mm")

if __name__ == "__main__":
    analyze_good_detections() 