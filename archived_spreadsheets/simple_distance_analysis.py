import numpy as np
import matplotlib.pyplot as plt

def analyze_exuviae_distance_impact():
    """
    Analyze how distance affects scaling factors for exuviae using the trigonometric formula.
    Using the specific heights: 680mm and 390mm
    """
    
    # Camera parameters (typical values)
    image_width = 1920
    image_height = 1080
    horizontal_fov = 60  # degrees
    vertical_fov = 40   # degrees
    
    # Exuviae specific heights
    height_680 = 680  # mm
    height_390 = 390  # mm
    
    # Distance range (from close to far)
    distances = np.linspace(500, 2000, 30)  # 500mm to 2000mm
    
    print("=== Exuviae Distance Impact Analysis ===")
    print(f"Camera FOV: {horizontal_fov}° horizontal, {vertical_fov}° vertical")
    print(f"Image resolution: {image_width}x{image_height}")
    print(f"Exuviae heights: {height_680}mm and {height_390}mm")
    print()
    
    # Calculate scaling factors for different distances and heights
    results = []
    
    for distance in distances:
        # Calculate scaling factors using the formula
        fov_x_rad = np.radians(horizontal_fov)
        fov_y_rad = np.radians(vertical_fov)
        
        # For height 680mm
        scale_x_680 = (2 * distance * np.tan(fov_x_rad / 2)) / image_width
        scale_y_680 = (2 * distance * np.tan(fov_y_rad / 2)) / image_height
        
        # For height 390mm  
        scale_x_390 = (2 * distance * np.tan(fov_x_rad / 2)) / image_width
        scale_y_390 = (2 * distance * np.tan(fov_y_rad / 2)) / image_height
        
        # Calculate combined scale for different angles (0°, 45°, 90°)
        angles = [0, 45, 90]
        
        for angle in angles:
            angle_rad = np.radians(angle)
            
            # Combined scale for height 680mm
            combined_scale_680 = np.sqrt((scale_x_680 * np.cos(angle_rad)) ** 2 + 
                                       (scale_y_680 * np.sin(angle_rad)) ** 2)
            
            # Combined scale for height 390mm
            combined_scale_390 = np.sqrt((scale_x_390 * np.cos(angle_rad)) ** 2 + 
                                       (scale_y_390 * np.sin(angle_rad)) ** 2)
            
            results.append({
                'distance_mm': distance,
                'height_mm': 680,
                'angle_deg': angle,
                'scale_x': scale_x_680,
                'scale_y': scale_y_680,
                'combined_scale': combined_scale_680
            })
            
            results.append({
                'distance_mm': distance,
                'height_mm': 390,
                'angle_deg': angle,
                'scale_x': scale_x_390,
                'scale_y': scale_y_390,
                'combined_scale': combined_scale_390
            })
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scaling factors vs distance for both heights
    ax1 = axes[0, 0]
    for height in [680, 390]:
        for angle in [0, 45, 90]:
            data = [r for r in results if r['height_mm'] == height and r['angle_deg'] == angle]
            distances_plot = [r['distance_mm'] for r in data]
            scales_plot = [r['combined_scale'] for r in data]
            ax1.plot(distances_plot, scales_plot, 
                    label=f'Height {height}mm, {angle}°', 
                    marker='o', markersize=3)
    
    ax1.set_xlabel('Distance (mm)')
    ax1.set_ylabel('Combined Scaling Factor (mm/pixel)')
    ax1.set_title('Scaling Factor vs Distance (Exuviae Heights)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Distance sensitivity (how much scaling changes with distance)
    ax2 = axes[0, 1]
    base_distance = 1000  # mm
    base_scale_x_680 = (2 * base_distance * np.tan(np.radians(horizontal_fov) / 2)) / image_width
    base_scale_x_390 = (2 * base_distance * np.tan(np.radians(vertical_fov) / 2)) / image_height
    
    sensitivity_680 = []
    sensitivity_390 = []
    
    for distance in distances:
        scale_x_680 = (2 * distance * np.tan(np.radians(horizontal_fov) / 2)) / image_width
        scale_x_390 = (2 * distance * np.tan(np.radians(vertical_fov) / 2)) / image_height
        
        sens_680 = (scale_x_680 - base_scale_x_680) / base_scale_x_680 * 100
        sens_390 = (scale_x_390 - base_scale_x_390) / base_scale_x_390 * 100
        
        sensitivity_680.append(sens_680)
        sensitivity_390.append(sens_390)
    
    ax2.plot(distances, sensitivity_680, 'b-', linewidth=2, label='Height 680mm')
    ax2.plot(distances, sensitivity_390, 'r-', linewidth=2, label='Height 390mm')
    ax2.set_xlabel('Distance (mm)')
    ax2.set_ylabel('Scaling Factor Change (%)')
    ax2.set_title('Distance Sensitivity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 3. Error magnification factor
    ax3 = axes[1, 0]
    # Show how a 1-pixel error translates to mm error at different distances
    pixel_error = 1  # 1 pixel error
    
    mm_error_680 = []
    mm_error_390 = []
    
    for distance in distances:
        scale_x_680 = (2 * distance * np.tan(np.radians(horizontal_fov) / 2)) / image_width
        scale_x_390 = (2 * distance * np.tan(np.radians(vertical_fov) / 2)) / image_height
        
        mm_error_680.append(pixel_error * scale_x_680)
        mm_error_390.append(pixel_error * scale_x_390)
    
    ax3.plot(distances, mm_error_680, 'b-', linewidth=2, label='Height 680mm')
    ax3.plot(distances, mm_error_390, 'r-', linewidth=2, label='Height 390mm')
    ax3.set_xlabel('Distance (mm)')
    ax3.set_ylabel('1-Pixel Error (mm)')
    ax3.set_title('Pixel Error Magnification')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scaling factor ratio between heights
    ax4 = axes[1, 1]
    scale_ratio = []
    for distance in distances:
        scale_x_680 = (2 * distance * np.tan(np.radians(horizontal_fov) / 2)) / image_width
        scale_x_390 = (2 * distance * np.tan(np.radians(vertical_fov) / 2)) / image_height
        ratio = scale_x_680 / scale_x_390
        scale_ratio.append(ratio)
    
    ax4.plot(distances, scale_ratio, 'g-', linewidth=2)
    ax4.set_xlabel('Distance (mm)')
    ax4.set_ylabel('Scale Ratio (680mm/390mm)')
    ax4.set_title('Scaling Factor Ratio Between Heights')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('exuviae_distance_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key insights
    print("=== Key Insights ===")
    print("1. Scaling factors are directly proportional to distance")
    print("2. Longer distances = larger scaling factors = larger measurement errors")
    print("3. Height differences affect scaling factors")
    print("4. 1-pixel error becomes larger in mm as distance increases")
    
    # Calculate specific examples
    print("\n=== Specific Examples ===")
    for distance in [800, 1200, 1600]:
        scale_x_680 = (2 * distance * np.tan(np.radians(horizontal_fov) / 2)) / image_width
        scale_x_390 = (2 * distance * np.tan(np.radians(vertical_fov) / 2)) / image_height
        
        print(f"Distance {distance}mm:")
        print(f"  Height 680mm: 1 pixel = {scale_x_680:.3f} mm")
        print(f"  Height 390mm: 1 pixel = {scale_x_390:.3f} mm")
        print(f"  Ratio: {scale_x_680/scale_x_390:.2f}")
        print()
    
    return results

if __name__ == "__main__":
    results = analyze_exuviae_distance_impact() 