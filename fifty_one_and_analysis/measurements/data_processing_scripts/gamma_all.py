
# Import the necessary packages
import numpy as np
import cv2
import glob
import os
import tqdm

def adjust_gamma(image, gamma=2.2):
    """
    Adjusts the gamma of an image using gamma correction.

    Parameters:
    image (numpy.ndarray): The input image.
    gamma (float): The gamma value for gamma correction. Default is 2.2.

    Returns:
    numpy.ndarray: The gamma-corrected image.
    """
    
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    gamma_corrected = cv2.LUT(image, table)

    return gamma_corrected

# adjust gamma for a folder of images

def adjust_gamma_folder(folder, gamma=2.2):
    """
    Apply gamma correction to all images in a folder.

    Args:
        folder (str): The path to the folder containing the images.
        gamma (float, optional): The gamma value for gamma correction. Defaults to 2.2.
    """
    # get all the files in the folder
    files = glob.glob(folder + '/*')
    # iterate over the files
    for file in tqdm.tqdm(files):
        # read the image
        img = cv2.imread(file)
        # apply gamma correction
        gamma_corrected = adjust_gamma(img, gamma)
        # get the original file name
        file_name = os.path.basename(file)
        # add 'gamma' to the file name
        new_file_name = file_name+ '_gamma.jpg'
        # save the image with the new file name
        folder_to_save = os.path.join(folder, 'gamma')   
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        cv2.imwrite(os.path.join(folder_to_save, new_file_name), gamma_corrected)
folder_path = "path/to/your/folder"
adjust_gamma_folder(folder_path, 2.2)