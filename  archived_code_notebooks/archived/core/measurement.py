class PrawnMeasurement:
    """Core class for prawn measurements using camera parameters and keypoints"""
    def __init__(self, image_width, image_height, horizontal_fov, vertical_fov, distance_mm):
        self.camera = CameraParameters(
            image_width, image_height,
            horizontal_fov, vertical_fov,
            distance_mm
        )
        
    def compute_length(self, keypoints_dict):
        """Calculate lengths between keypoints
        Returns dictionary of measurements:
        - total_length: tail to rostrum
        - carapace_length: start_carapace to rostrum
        """
        measurements = {}
        
        # Calculate total length (tail to rostrum)
        tail = keypoints_dict['tail']
        rostrum = keypoints_dict['rostrum']
        total_length, angle = self.camera.compute_real_distance(tail, rostrum)
        measurements['total_length'] = total_length
        
        # Calculate carapace length (start_carapace to rostrum)
        start_carapace = keypoints_dict['start_carapace']
        carapace_length, angle = self.camera.compute_real_distance(start_carapace, rostrum)
        measurements['carapace_length'] = carapace_length
        
        return measurements 