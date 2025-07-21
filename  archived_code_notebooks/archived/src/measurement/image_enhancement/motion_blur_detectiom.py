import cv2
import numpy as np
import cv2
import numpy as np
from tqdm import tqdm
import glob
import os

def detect_motion_blur(frame, variance_threshold=600):
    """
    Detects motion blur in a given frame.

    Parameters:
    - frame: The input frame to be analyzed.
    - variance_threshold: The threshold value for variance of the magnitude spectrum.
                          If the variance is below this threshold, it might indicate motion blur.

    Returns:
    - is_blur: A boolean value indicating whether motion blur is detected or not.
    - variance: The variance of the magnitude spectrum.

    """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute the Fourier Transform of the image
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Compute the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # Adding 1 to avoid log(0)
    
    # Calculate the variance of the magnitude spectrum
    variance = np.var(magnitude_spectrum)
    
    # If variance is below the threshold, it might indicate motion blur
    if variance < variance_threshold:
        return True, variance
    else:
        return False, variance

# Replace 'path_to_your_video.mp4' with the actual path to your video file


def process_images_folder(folder, variance_threshold=550):
    """
    Process images in a folder and detect motion blur.

    Args:
        folder (str): The path to the folder containing the images.
        variance_threshold (int, optional): The threshold for detecting motion blur. Defaults to 550.

    Returns:
        None
    """
    # get all the files in the folder
    files = glob.glob(folder + '/*')
    # iterate over the files
    folder_to_save = os.path.join('./src/image_enhancement', 'blur')
    for file in tqdm(files):
        # read the image
        img = cv2.imread(file)
        #check if img is image file
        if img is None:
            continue
        # apply gamma correction
        is_blurry, variance = detect_motion_blur(img, variance_threshold)
        new_file_name = os.path.basename(file)+ '_blur.jpg'
        if is_blurry:
            print(f"{file} is blurry with variance {variance}")
            cv2.imwrite(os.path.join(folder_to_save, new_file_name), img)
        else:
            print(f"{file} is not blurry with variance {variance}")
process_images_folder("./src/image_enhancement/valid_gamma", 525)