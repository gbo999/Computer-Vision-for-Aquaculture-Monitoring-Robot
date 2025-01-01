class CameraParameters:
    """Handles camera-specific calculations and parameters"""
    def __init__(self, image_width, image_height, h_fov, v_fov, distance_mm):
        self.image_width = image_width
        self.image_height = image_height
        self.h_fov = h_fov
        self.v_fov = v_fov
        self.distance_mm = distance_mm
        self.scale_x, self.scale_y = self._calculate_scaling_factors()

    def _calculate_scaling_factors(self):
        """Calculate mm/pixel scaling based on FOV and distance""" 