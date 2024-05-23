import numpy as np

def crop_images(images, image_data, x_size=497, y_size=365):

    """
    Crop images based on specified dimensions and center-specific cropping requirements.

    Args:
    - images (dict): Dictionary containing image data with file names as keys.
    - image_data (DataFrame): DataFrame containing metadata associated with the images.
    - x_size (int, optional): Width of the cropped images. Defaults to 497 (obtained from training data).
    - y_size (int, optional): Height of the cropped images. Defaults to 365 (obtained from training data).

    Returns:
    - cropped_images (dict): Dictionary containing cropped image data with file names as keys.

    This function crops images based on specified dimensions and center-specific cropping requirements.
    It iterates over each center in the metadata and applies cropping accordingly to each image associated with that center.
    Cropping dimensions are adjusted based on the center, and the resulting cropped images are stored in a dictionary.
    """

    centers=set(image_data['Center'])

    cropped_images={}

    for i, center in enumerate(centers):
        image_data_center = image_data[image_data['Center'] == center]
        keys_center= list(image_data_center.iloc[:]['Anonymized ID'])


        for k in keys_center:

            if "Ljubljana"in center:
                im=images[k][:, 109:823]
            elif "Milano"in center:
                im=images[k][:400, 65:562]
            elif "Udine"in center:
                im=images[k][16:393, 8:910]
                im=im[:510,:]
            else:
                im=images[k]

            center_x=im.shape[1]//2
            img_array=im[:y_size,center_x-x_size//2:center_x+x_size//2 ]
            cropped_images[k] = np.stack((img_array,) * 3, axis=-1)

    return cropped_images

def gland_type_division(images,image_data):

    """
    Divide images into submandibular and parotid types based on metadata.

    Args:
    - images (dict): Dictionary containing image data with file names as keys.
    - image_data (DataFrame): DataFrame containing metadata associated with the images.

    Returns:
    - submandibular_images (list): List of submandibular gland images.
    - submandibular_labels (list): List of labels corresponding to submandibular gland images.
    - parotid_images (list): List of parotid gland images.
    - parotid_labels (list): List of labels corresponding to parotid gland images.

    This function iterates over the image metadata and divides the images into two types: submandibular and parotid.
    It extracts images and associated labels based on the gland type specified in the metadata.
    The function returns lists of submandibular and parotid gland images along with their respective labels.
    """

    submandibular_images = []
    submandibular_labels=[]
    parotid_images = []
    parotid_labels=[]

    for _, row in image_data.iterrows():

        anonymized_id = row['Anonymized ID']
        image = images.get(anonymized_id)
        omeract_score = row['OMERACT score']
        
        if row['parotid/submandibular'] == 'parotid':
            parotid_images.append(image)
            parotid_labels.append(omeract_score)
        elif row['parotid/submandibular'] == 'submandibular':
            submandibular_images.append(image)
            submandibular_labels.append(omeract_score)
    

        return submandibular_images, submandibular_labels, parotid_images, parotid_labels
