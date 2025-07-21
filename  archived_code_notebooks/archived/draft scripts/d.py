import cv2
import numpy as np

def recolor_shrimp_and_background(
    image_path,
    lower_shrimp=(5, 50, 50),
    upper_shrimp=(20, 255, 255),
    shrimp_bgr=(42, 42, 165),      # BGR for "brownish"
    background_bgr=(208, 224, 64) # BGR for "turquoise"
):
    """
    Reads an image, creates a mask for the shrimp color range (HSV),
    and recolors the shrimp to brown and the rest to turquoise.

    :param image_path: Path to the input image file
    :param lower_shrimp: Lower HSV threshold for shrimp color
    :param upper_shrimp: Upper HSV threshold for shrimp color
    :param shrimp_bgr: BGR color tuple for the shrimp
    :param background_bgr: BGR color tuple for the background
    :return: A BGR image with shrimp in brown and background in turquoise
    """
    # 1. Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read the image from {image_path}")

    # 2. Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Create a mask for the shrimp
    mask = cv2.inRange(hsv, np.array(lower_shrimp), np.array(upper_shrimp))

    # 4. Create an output image (same size as original)
    result = np.zeros_like(img, dtype=np.uint8)

    # 5. Where mask=255 (shrimp), assign brown; otherwise turquoise
    result[mask == 255] = shrimp_bgr
    result[mask == 0] = background_bgr

    return result

def shift_hsv(image, hue_shift, sat_shift, val_shift):
    """
    Shift the hue, saturation, and value (brightness) of an image in HSV space.

    :param image: Input BGR image (NumPy array).
    :param hue_shift: Amount to add to the hue channel (OpenCV hue range: 0..179).
    :param sat_shift: Amount to add to the saturation channel (range 0..255).
    :param val_shift: Amount to add to the value (brightness) channel (range 0..255).
    :return: The color-shifted BGR image.
    """
    # Convert from BGR to HSV (OpenCV hue: 0..179, sat/value: 0..255)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)

    # 1. Shift Hue with wrap-around
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180

    # 2. Shift Saturation (clip to [0..255])
    hsv[..., 1] = np.clip(hsv[..., 1] + sat_shift, 0, 255)

    # 3. Shift Value (clip to [0..255])
    hsv[..., 2] = np.clip(hsv[..., 2] + val_shift, 0, 255)

    # Convert back to uint8
    hsv = hsv.astype(np.uint8)

    # Convert from HSV back to BGR
    shifted_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return shifted_img

if __name__ == "__main__":
    # -----------------------------
    # Example usage
    # -----------------------------
    input_path = r"/Users/gilbenor/Library/CloudStorage/OneDrive-Personal/measurement_paper_images/molt/all molt/undistorted/resized/undistorted_GX010194_26_373.jpg"   # <-- Replace with your input image path
    output_path = "output.jpg" # <-- Desired output path

    # Step 1: Recolor shrimp to brown and background to turquoise
    recolored_img = recolor_shrimp_and_background(
        image_path=input_path,
        lower_shrimp=(5, 50, 50),    # Adjust to isolate your shrimp color
        upper_shrimp=(20, 255, 255)  # Adjust to isolate your shrimp color
    )

    # Step 2: Apply HSV shifts: hue = -5, saturation = +61, value = -61
    final_img = shift_hsv(recolored_img, hue_shift=-5, sat_shift=61, val_shift=-61)

    # Step 3: Save the final result
    cv2.imwrite(output_path, final_img)
    print(f"Saved final output to {output_path}")
