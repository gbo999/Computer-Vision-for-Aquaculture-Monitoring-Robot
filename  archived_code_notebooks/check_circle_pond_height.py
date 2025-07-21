import math

class ObjectLengthMeasurer:
    def __init__(self, image_width, image_height, horizontal_fov, vertical_fov, distance_mm):
        self.image_width = image_width
        self.image_height = image_height
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.distance_mm = distance_mm
        self.scale_x, self.scale_y = self.calculate_scaling_factors()

    def calculate_scaling_factors(self):
        """
        Calculate the scaling factors (mm per pixel) based on the camera's FOV and distance.
        """
        fov_x_rad = math.radians(self.horizontal_fov)
        fov_y_rad = math.radians(self.vertical_fov)
        scale_x = (2 * self.distance_mm * math.tan(fov_x_rad / 2)) / self.image_width
        scale_y = (2 * self.distance_mm * math.tan(fov_y_rad / 2)) / self.image_height
        return scale_x, scale_y

def check_circle_pond_height():
    """
    Check how different height values affect scaling factors for Circle Pond.
    """
    
    # Camera parameters from the data_loader.py
    image_width = 5312
    image_height = 2988
    horizontal_fov = 75.2
    vertical_fov = 46
    
    print("=== Circle Pond Height Impact Analysis ===")
    print(f"Camera: {image_width}x{image_height}, FOV: {horizontal_fov}°x{vertical_fov}°")
    print()
    
    # Current heights
    circle_current = 680  # mm
    square_current = 390  # mm
    
    print(f"Current heights:")
    print(f"  Circle Pond: {circle_current}mm")
    print(f"  Square Pond: {square_current}mm")
    print()
    
    # Calculate current scaling factors
    circle_measurer = ObjectLengthMeasurer(image_width, image_height, horizontal_fov, vertical_fov, circle_current)
    square_measurer = ObjectLengthMeasurer(image_width, image_height, horizontal_fov, vertical_fov, square_current)
    
    print(f"Current scaling factors:")
    print(f"  Circle Pond: scale_x={circle_measurer.scale_x:.6f}, scale_y={circle_measurer.scale_y:.6f}")
    print(f"  Square Pond: scale_x={square_measurer.scale_x:.6f}, scale_y={square_measurer.scale_y:.6f}")
    print()
    
    # Check different heights for Circle Pond (600-800mm in 20mm increments)
    print("=== Different Heights for Circle Pond ===")
    heights = range(600, 801, 20)
    
    for height in heights:
        measurer = ObjectLengthMeasurer(image_width, image_height, horizontal_fov, vertical_fov, height)
        
        # Calculate percentage difference from current Circle Pond
        scale_diff_x = ((measurer.scale_x - circle_measurer.scale_x) / circle_measurer.scale_x) * 100
        scale_diff_y = ((measurer.scale_y - circle_measurer.scale_y) / circle_measurer.scale_y) * 100
        
        print(f"Height {height}mm: scale_x={measurer.scale_x:.6f}, scale_y={measurer.scale_y:.6f}")
        print(f"  Difference from current: {scale_diff_x:+.1f}% (x), {scale_diff_y:+.1f}% (y)")
        
        # Check if this height would be closer to Square Pond scaling
        square_diff_x = abs(measurer.scale_x - square_measurer.scale_x)
        current_diff_x = abs(circle_measurer.scale_x - square_measurer.scale_x)
        
        if square_diff_x < current_diff_x:
            print(f"  *** This height brings Circle Pond closer to Square Pond scaling! ***")
        print()
    
    # Calculate what height would give Circle Pond the same scaling as Square Pond
    print("=== Height to Match Square Pond Scaling ===")
    target_scale_x = square_measurer.scale_x
    target_height_x = (target_scale_x * image_width) / (2 * math.tan(math.radians(horizontal_fov) / 2))
    
    target_scale_y = square_measurer.scale_y
    target_height_y = (target_scale_y * image_height) / (2 * math.tan(math.radians(vertical_fov) / 2))
    
    print(f"To match Square Pond scale_x ({target_scale_x:.6f}): height = {target_height_x:.1f}mm")
    print(f"To match Square Pond scale_y ({target_scale_y:.6f}): height = {target_height_y:.1f}mm")
    print(f"Average target height: {(target_height_x + target_height_y) / 2:.1f}mm")
    print()
    
    print("=== Conclusion ===")
    print("If Circle Pond height measurement is wrong, it could cause:")
    print("1. Incorrect scaling factors")
    print("2. Artificial compensation for pixel errors")
    print("3. ρ = 0.30 (artificial compensation) instead of realistic scaling")
    print()
    print("The current Circle Pond height (680mm) might be incorrect!")

if __name__ == "__main__":
    check_circle_pond_height() 