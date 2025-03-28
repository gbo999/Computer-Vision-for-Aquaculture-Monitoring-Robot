import numpy as np


def normalize_angle(angle):
        """
        Normalize the angle to [0°, 90°].
        """
        theta_norm = min(abs(angle % 180), 180 - abs(angle % 180))

        return  theta_norm

def test_angle_normalization():
    # Create an instance of the class containing normalize_angle
    
    # Test cases with different angle ranges
    test_angles = [
        # Basic cases
        0, 45, 90, 180, 270, 360,
        
        # Negative angles
        -45, -90, -180, -270, -360,
        
        # Decimal angles
        22.5, 135.7, -67.3,
        
        # Large angles
        720, -720, 1080,
        
        # Edge cases
        0.001, 179.999, -0.001
    ]
    
    print("Testing angle normalization:")
    print("-" * 50)
    print(f"{'Input Angle':>15} | {'Normalized Angle':>15}")
    print("-" * 50)
    
    for angle in test_angles:
        normalized = normalize_angle(angle)
        print(f"{angle:>15.2f} | {normalized:>15.2f}")
    
    # Visual representation using matplotlib
    import matplotlib.pyplot as plt
    
    # Generate a smooth range of angles for plotting
    input_angles = np.linspace(-360, 360, 1000)
    normalized_angles = [normalize_angle(angle) for angle in input_angles]
    
    plt.figure(figsize=(12, 6))
    plt.plot(input_angles, normalized_angles, 'b-', label='Normalized angle')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input angle (degrees)')
    plt.ylabel('Normalized angle (degrees)')
    plt.title('Angle Normalization Function Behavior')
    plt.legend()
    
    # Add reference lines
    plt.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90° line')
    plt.axhline(y=180, color='g', linestyle='--', alpha=0.5, label='180° line')
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_angle_normalization()