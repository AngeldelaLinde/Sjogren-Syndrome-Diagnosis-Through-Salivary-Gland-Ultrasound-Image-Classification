# Sjogren Syndrome Diagnosis through Salivary Gland Ultrasonography (SGUS) Image Classification

This repository contains code for diagnosing Sjogren Syndrome through the classification of salivary gland ultrasonography (SGUS) images using deep learning models. The main focus is on the preprocessing, training, and evaluation of the models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Data Loading and Visualization](#data-loading-and-visualization)
  - [Image Preprocessing](#image-preprocessing)
  - [Dataset Splitting](#dataset-splitting)
  - [Dataloader Creation](#dataloader-creation)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

Clone this repository:
    ```sh
    git clone https://github.com/your-username/sjogren-syndrome-diagnosis.git
    ```

## Usage

### Data Loading and Visualization

Load the dataset and visualize the images by centers:
```python
from load_dataset import load_dataset
import matplotlib.pyplot as plt

file_path = "./Imágenes + Labels"
images_path = file_path
csv_path = file_path + "/Anonymized images_student.xlsx"
images, image_data = load_dataset(images_path, csv_path)

centers = set(image_data['Center'])
fig, axes = plt.subplots(nrows=len(centers), ncols=3, figsize=(15, 5 * len(centers)))
for i, center in enumerate(centers):
    image_data_center = image_data[image_data['Center'] == center]
    for j in range(3):
        keys_example = image_data_center.iloc[j]['Anonymized ID']
        im_example = images[keys_example]
        axes[i, j].imshow(im_example, cmap='gray')
        axes[i, j].set_title(f'Center: {center}\n Anonymized ID: {keys_example}')
plt.tight_layout()
plt.show()
