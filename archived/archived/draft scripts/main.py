def main():
    # Initialize components
    dataset = PrawnDataset()
    detector = PrawnDetector()
    measurement = PrawnMeasurement(
        image_width=5312,
        image_height=2988,
        horizontal_fov=75.2,
        vertical_fov=46,
        distance_mm=1000
    )
    metrics = MeasurementMetrics()

    # Process images
    for image_path in image_paths:
        # Detect prawns and keypoints
        predictions = detector.process_image(image_path)
        
        # Calculate measurements
        lengths = measurement.compute_length(predictions.keypoints)
        
        # Calculate metrics
        errors = metrics.calculate_errors(lengths, ground_truth)
        
        # Add to dataset
        dataset.add_sample(image_path, predictions, errors)

    # Visualize results
    dataset.launch_viewer() 