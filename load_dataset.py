import pandas as pd
import os
import matplotlib.pyplot as plt

def load_dataset(images_path,csv_path):
    """
    Load dataset consisting of images and associated metadata from Excel file.
    
    Args:
    - images_path (str): Path to the directory containing image files.
    - csv_path (str): Path to the Excel file containing metadata.
    
    Returns:
    - images (dict): Dictionary containing image data with file names as keys.
    - image_data (DataFrame): DataFrame containing metadata associated with the images.
    
    This function reads image files from the specified directory and loads metadata from an Excel file.
    It matches image data with corresponding metadata based on file names and returns the images as a dictionary
    with keys representing file names and associated metadata in a DataFrame.
    """
    image_data = pd.read_excel(csv_path)

    images={}
    file_list = os.listdir(images_path)
    jpg_files = [file for file in file_list if file.lower().endswith('.jpg')]


    for file_name in jpg_files:
        im_path = os.path.join(images_path, file_name)
        image = plt.imread(im_path)
        file_key = int(os.path.splitext(file_name)[0])

        # Store the image data in the dictionary with the file name as the key
        images[file_key] = image

    keys = sorted(images.keys(), key=lambda x: int(x)) #Key is eqivalent to Anonymized ID in patient_data
    images = {key: images[key] for key in keys}

    return images, image_data
