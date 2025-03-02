import pandas as pd
import numpy as np
import os
import math
from pathlib import Path

def calculate_euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def determine_size(total_length_mm):
    """
    Determine if a detection is big or small based on total length only.
    Big: 175-220mm          
    Small: 116-174mm
    """
    # Big: fixed range
    if 175 <= total_length_mm <= 220:
        return "BIG"
    
    # Small: percentage range
    small_expected = 145
    small_min = small_expected * 0.8  # -20%
    small_max = small_expected * 1.2  # +20%
    if small_min <= total_length_mm <= small_max:
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
    labels_dir = Path('runs/pose/predict57/filtered_labels')
    
    # Image dimensions for calculations
    calc_width = 5312  # Original image width
    calc_height = 2988  # Original image height
    
    # Process each good image
    for _, row in good_images.iterrows():
        image_name = row['image_name']
        base_name = image_name.replace('viz_', '').replace('viz_segmented_', '')
        
        # Find corresponding label file - add "segmented_" prefix
        label_file = labels_dir / f"segmented_{base_name.replace('.jpg', '.txt')}"
        if not label_file.exists():
            print(f"Warning: Label file not found for {image_name}")
            continue
        
        # Initialize entry with all required fields
        entry = {
            'image_name': base_name,
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
        is_circle2 = "GX010191" in base_name
        height_mm = 700 if is_circle2 else 410
        
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
                diagonal_image_size = math.sqrt(calc_width ** 2 + calc_height ** 2)
                total_length_mm = (total_length_pixels / diagonal_image_size) * (2 * height_mm * math.tan(math.radians(84.6/2)))
                carapace_length_mm = (carapace_length_pixels / diagonal_image_size) * (2 * height_mm * math.tan(math.radians(84.6/2)))
                
                # Determine size
                size = determine_size(total_length_mm)
                further_labels_dir = Path('runs/pose/predict57/further_labels_files')
                further_labels_dir.mkdir(exist_ok=True)
                # Store measurements in the appropriate columns
                if size == "BIG":
                    entry['big_total_length'] = round(total_length_mm, 1)
                    entry['big_carapace_length'] = round(carapace_length_mm, 1)
                    entry['big_eye_x'] = round(eye_x, 1)
                    entry['big_eye_y'] = round(eye_y, 1)
                    
                    #add the full detection to the further label file
                    image_name = Path(entry['image_name']).stem
                    label_file = further_labels_dir / f"{image_name}.txt"
                    with open(label_file, 'a') as f:
                        f.write(detection)
                elif size == "SMALL":
                    entry['small_total_length'] = round(total_length_mm, 1)
                    entry['small_carapace_length'] = round(carapace_length_mm, 1)
                    entry['small_eye_x'] = round(eye_x, 1)
                    entry['small_eye_y'] = round(eye_y, 1)

                    #add the full detection to the further label file
                    image_name = Path(entry['image_name']).stem
                    label_file = further_labels_dir / f"{image_name}.txt"
                    with open(label_file, 'a') as f:
                        f.write(detection)
                    
                    
                    # Create label file path with same name as image
                   

                        

                    #add detection to  fuether label file
        

        # Add to analysis data
        analysis_data.append(entry)
    
    # Create DataFrame and save to CSV
    analysis_df = pd.DataFrame(analysis_data)
    output_file = 'runs/pose/predict57/length_analysis.csv'
    analysis_df.to_csv(output_file, index=False)
    print(f"Analysis saved to {output_file}")
    
    # Print summary statistics
    print(f"\nTotal good images analyzed: {len(analysis_data)}")
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