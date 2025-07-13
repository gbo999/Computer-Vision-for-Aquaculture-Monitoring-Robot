import random
import pandas as pd
import fiftyone as fo
import os
import numpy as np

def create_measurement_dataset():
    """
    Create a FiftyOne dataset from the merged CSV showing:
    - Total length and carapace length measurements
    - Big/small classifications  
    - Real-world reference lengths (180mm, 63mm for big; 145mm, 41mm for small)
    """
    
    # Load the merged data
    merged_df = pd.read_csv("spreadsheet_files/merged_manual_shai_keypoints.csv")
    print(f"Loaded {len(merged_df)} measurements from merged CSV")
    
    # Images directory
    images_dir = "/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized"
    
    # Delete existing dataset if it exists
    dataset_name = "prawn_measurements"
    if fo.dataset_exists(dataset_name):
        print(f"Deleting existing dataset '{dataset_name}'...")
        fo.delete_dataset(dataset_name)
    
    # Create new dataset
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    
    # Define skeleton for keypoints
    skeleton = fo.KeypointSkeleton(
        labels=["start_carapace", "eyes", "rostrum", "tail"],
        edges=[[0, 1], [1, 2], [2, 3]]  # Connect: start_carapace->eyes->rostrum->tail
    )
    dataset.default_skeleton = skeleton
    
    # Reference lengths for big and small prawns
    BIG_TOTAL_REF = 180  # mm
    BIG_CARAPACE_REF = 63  # mm
    SMALL_TOTAL_REF = 145  # mm
    SMALL_CARAPACE_REF = 41  # mm
    
    # Process each unique image
    processed_images = set()
    
    for _, row in merged_df.iterrows():
        image_name = row['image_name']
        
        # Skip if we've already processed this image
        if image_name in processed_images:
            continue
            
        processed_images.add(image_name)
        
        # Construct image path
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
            
        # Create FiftyOne sample
        sample = fo.Sample(filepath=image_path)
        
        # Get all measurements for this image
        image_measurements = merged_df[merged_df['image_name'] == image_name]
        
        # Create detections for each measurement
        detections = []
        keypoints_list = []
        
        # Track separate objects for each measurement
        shai_objects = []  # Will store individual Shai detection objects
        polyline_objects = []  # Will store individual polyline objects
        
        for _, measurement in image_measurements.iterrows():
            # Get bounding box (use manual bbox)
            bbox = [
                measurement['manual_bbox_x'],
                measurement['manual_bbox_y'], 
                measurement['manual_bbox_width'],
                measurement['manual_bbox_height']
            ]
            
            # Skip if bbox is invalid
            if any(np.isnan(coord) for coord in bbox):
                continue
                
            # Get size classification
            size = measurement['manual_size']
            
            # Get measurements
            total_length = measurement['calculated_total_length_mm']
            carapace_length = measurement['calculated_carapace_length_mm']
            shai_length_pixels = measurement['shai_length']  # Keep in pixels
            
            # Image dimensions
            img_width_px = 5312
            img_height_px = 2988
            
            # Calculate total length in pixels from keypoints (if available)
            total_length_pixels = None
            if (not np.isnan(measurement['rostrum_x']) and not np.isnan(measurement['rostrum_y']) and 
                not np.isnan(measurement['tail_x']) and not np.isnan(measurement['tail_y'])):
                # Convert normalized coordinates to pixels
                rostrum_x_px = measurement['rostrum_x'] * img_width_px
                rostrum_y_px = measurement['rostrum_y'] * img_height_px
                tail_x_px = measurement['tail_x'] * img_width_px
                tail_y_px = measurement['tail_y'] * img_height_px
                total_length_pixels = np.sqrt((rostrum_x_px - tail_x_px)**2 + (rostrum_y_px - tail_y_px)**2)
            
            # Get reference lengths based on size
            if size == 'big':
                total_ref = BIG_TOTAL_REF
                carapace_ref = BIG_CARAPACE_REF
            else:  # small
                total_ref = SMALL_TOTAL_REF
                carapace_ref = SMALL_CARAPACE_REF
            
            # Calculate errors from reference
            total_error = None
            carapace_error = None
            total_error_pct = None
            carapace_error_pct = None
            
            if not np.isnan(total_length):
                total_error = abs(total_length - total_ref)
                total_error_pct = (total_error / total_ref) * 100
                
            if not np.isnan(carapace_length):
                carapace_error = abs(carapace_length - carapace_ref)
                carapace_error_pct = (carapace_error / carapace_ref) * 100
            
            # Create label with all information
            label_parts = [
                f"Size: {size}",
                f"Total: {total_length:.1f}mm" if not np.isnan(total_length) else "Total: N/A",
                f"Carapace: {carapace_length:.1f}mm" if not np.isnan(carapace_length) else "Carapace: N/A",
                f"Shai: {shai_length_pixels:.1f}px",
                f"Total px: {total_length_pixels:.1f}px" if total_length_pixels is not None else "Total px: N/A",
                f"Ref Total: {total_ref}mm",
                f"Ref Carapace: {carapace_ref}mm"
            ]
            
            if total_error is not None:
                label_parts.append(f"Total Error: {total_error:+.1f}mm ({total_error_pct:+.1f}%)")
            if carapace_error is not None:
                label_parts.append(f"Carapace Error: {carapace_error:+.1f}mm ({carapace_error_pct:+.1f}%)")
                
            label = " | ".join(label_parts)
            
            # Create detection
            detection = fo.Detection(
                label=label,
                bounding_box=bbox,
                confidence=1.0,
                # Add custom fields
                size_classification=size,
                total_length_mm=float(total_length) if not np.isnan(total_length) else None,
                carapace_length_mm=float(carapace_length) if not np.isnan(carapace_length) else None,
                shai_length_pixels=float(shai_length_pixels),
                total_length_pixels=float(total_length_pixels) if total_length_pixels is not None else None,
                total_reference_mm=total_ref,
                carapace_reference_mm=carapace_ref,
                total_error_mm=float(total_error) if total_error is not None else None,
                carapace_error_mm=float(carapace_error) if carapace_error is not None else None,
                total_error_pct=float(total_error_pct) if total_error_pct is not None else None,
                carapace_error_pct=float(carapace_error_pct) if carapace_error_pct is not None else None,
                iou_with_shai=float(measurement['shai_iou']),
                keypoints_available=bool(measurement['keypoints_available']),
                keypoints_valid_count=int(measurement['keypoints_valid_total']) if not np.isnan(measurement['keypoints_valid_total']) else 0
            )
            
            detections.append(detection)
            
            # Create keypoints if available
            if measurement['keypoints_available']:
                # Create keypoints using the 4 points: start_carapace, eyes, rostrum, tail
                keypoint_coords = []
                keypoint_labels = ["start_carapace", "eyes", "rostrum", "tail"]
                
                for i, label in enumerate(keypoint_labels):
                    x_col = f"{label}_x"
                    y_col = f"{label}_y"
                    
                    if x_col in measurement and y_col in measurement:
                        x = measurement[x_col]
                        y = measurement[y_col]
                        
                        # Handle NaN values - FiftyOne expects (x, y) pairs
                        if np.isnan(x) or np.isnan(y):
                            keypoint_coords.append([np.nan, np.nan])  # Invalid keypoint
                        else:
                            keypoint_coords.append([x, y])  # Valid keypoint
                    else:
                        keypoint_coords.append([np.nan, np.nan])  # Missing keypoint
                
                # Create keypoint object
                keypoint = fo.Keypoint(
                    label=f"{size}_keypoints",
                    points=keypoint_coords
                )
                keypoints_list.append(keypoint)
            
            # Create Shai's bounding box detection (convert from pixels to normalized)
            shai_bbox = [
                measurement['shai_bbox_x'] / img_width_px,  # x normalized
                measurement['shai_bbox_y'] / img_height_px,  # y normalized  
                measurement['shai_bbox_width'] / img_width_px,  # width normalized
                measurement['shai_bbox_height'] / img_height_px  # height normalized
            ]
            
            # Skip if Shai's bbox is invalid
            if not any(np.isnan(coord) for coord in shai_bbox):
                # Create individual Shai detection object
                shai_detection = fo.Detections(detections=[
                    fo.Detection(
                        label=f"Shai: {shai_length_pixels:.1f}px",
                        bounding_box=shai_bbox,
                        confidence=1.0,
                        shai_measurement=True,
                        shai_length_pixels=float(shai_length_pixels)
                    )
                ])
                shai_objects.append(shai_detection)
                
                # Create diagonal polylines for Shai's bbox
                # Use normalized coordinates for polylines
                x, y, w, h = shai_bbox
                
                # Create individual polyline objects
                diagonal1 = fo.Polylines(polylines=[
                    fo.Polyline(
                        label=f"Shai_diag1: {shai_length_pixels:.1f}px",
                        points=[
                            [(x, y), (x + w, y + h)]  # Top-left to bottom-right
                        ],
                        closed=False,
                        filled=False
                    )
                ])
                
                diagonal2 = fo.Polylines(polylines=[
                    fo.Polyline(
                        label=f"Shai_diag2: {shai_length_pixels:.1f}px",
                        points=[
                            [(x + w, y), (x, y + h)]  # Top-right to bottom-left
                        ],
                        closed=False,
                        filled=False
                    )
                ])
                
                polyline_objects.extend([diagonal1, diagonal2])
        
        # Add all annotations to sample
        if detections:
            # Add manual detections
            sample["detections"] = fo.Detections(detections=detections)
            
            # Add keypoints
            if keypoints_list:
                sample["keypoints"] = fo.Keypoints(keypoints=keypoints_list)
                
            # Add each Shai detection as separate object
            for i, shai_obj in enumerate(shai_objects):
                sample[f"shai_detection_{i}"] = shai_obj
                
            # Add each polyline as separate object
            for i, polyline_obj in enumerate(polyline_objects):
                sample[f"polyline_{i}"] = polyline_obj
            
            # Add sample-level metadata
            sample_measurements = image_measurements.iloc[0]  # Get first row for sample metadata
            
            # Determine pond type
            pond_type = "Circle" if "GX010191" in image_name else "Square"
            sample["pond_type"] = pond_type
            sample["tags"] = [pond_type.lower()]
            
            # Add measurement counts
            big_count = len(image_measurements[image_measurements['manual_size'] == 'big'])
            small_count = len(image_measurements[image_measurements['manual_size'] == 'small'])
            sample["big_count"] = big_count
            sample["small_count"] = small_count
            sample["total_measurements"] = len(image_measurements)
            
            # Add to dataset
            dataset.add_sample(sample)
    
    # Save dataset
    dataset.save()
    
    print(f"\nDataset '{dataset_name}' created with {len(dataset)} samples")
    
    # Count different types of annotations
    total_detections = sum(len(sample.detections.detections) for sample in dataset if sample.detections)
    total_keypoints = sum(len(sample.keypoints.keypoints) for sample in dataset if hasattr(sample, 'keypoints') and sample.keypoints)
    
    # Count Shai detections and polylines (they're in separate fields)
    total_shai_detections = 0
    total_polylines = 0
    
    for sample in dataset:
        # Count Shai detection fields
        for field_name in sample.field_names:
            if field_name.startswith('shai_detection_'):
                shai_field = sample[field_name]
                if shai_field:
                    total_shai_detections += len(shai_field.detections)
            elif field_name.startswith('polyline_'):
                polyline_field = sample[field_name]
                if polyline_field:
                    total_polylines += len(polyline_field.polylines)
    
    print(f"Total manual detections: {total_detections}")
    print(f"Total Shai detections: {total_shai_detections}")
    print(f"Total keypoints: {total_keypoints}")
    print(f"Total polylines: {total_polylines}")
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    
    # Count by size and type
    manual_big_count = 0
    manual_small_count = 0
    circle_count = 0
    square_count = 0
    
    for sample in dataset:
        if sample.detections:
            for det in sample.detections.detections:
                if hasattr(det, 'size_classification'):
                    if det.size_classification == 'big':
                        manual_big_count += 1
                    else:
                        manual_small_count += 1
        
        if sample.pond_type == 'Circle':
            circle_count += 1
        else:
            square_count += 1
    
    print(f"Manual big prawns: {manual_big_count}")
    print(f"Manual small prawns: {manual_small_count}")
    print(f"Circle pond samples: {circle_count}")
    print(f"Square pond samples: {square_count}")
    
    # Create useful views
    big_view = dataset.filter_labels("detections", fo.ViewField("size_classification") == "big")
    small_view = dataset.filter_labels("detections", fo.ViewField("size_classification") == "small")
    circle_view = dataset.match(fo.ViewField("pond_type") == "Circle")
    square_view = dataset.match(fo.ViewField("pond_type") == "Square")
    
    print(f"\nUseful views created:")
    print(f"- Big prawns: {len(big_view)} samples")
    print(f"- Small prawns: {len(small_view)} samples")
    print(f"- Circle pond: {len(circle_view)} samples")
    print(f"- Square pond: {len(square_view)} samples")
    
    print(f"\nLaunching FiftyOne app...")
    print(f"Dataset contains:")
    print(f"- Manual detections with size classifications (big/small)")
    print(f"- Shai's detections with pixel measurements")
    print(f"- Keypoints (start_carapace, eyes, rostrum, tail)")
    print(f"- Polylines showing diagonals of Shai's bounding boxes")
    print(f"- Calculated total and carapace lengths in mm")
    print(f"- Total lengths in pixels for comparison")
    print(f"- Reference lengths (Big: {BIG_TOTAL_REF}mm total, {BIG_CARAPACE_REF}mm carapace)")
    print(f"- Reference lengths (Small: {SMALL_TOTAL_REF}mm total, {SMALL_CARAPACE_REF}mm carapace)")
    print(f"- Error measurements from reference values")
    print(f"- IoU scores between manual and Shai's bounding boxes")
    
    # Launch FiftyOne
    session = fo.launch_app(dataset,port=random.randint(5000,6000))
    session.wait()

if __name__ == "__main__":
    create_measurement_dataset() 