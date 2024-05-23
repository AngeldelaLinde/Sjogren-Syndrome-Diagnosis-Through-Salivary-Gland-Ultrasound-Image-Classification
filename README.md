# Sjogren Syndrome Diagnosis through Salivary Gland Ultrasonography (SGUS) Image Classification

This repository contains the code and resources for diagnosing Sjogren Syndrome using deep learning techniques to classify salivary gland ultrasonography (SGUS) images. The workflow includes data loading, preprocessing, model training, and evaluation.

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

To install the necessary dependencies, clone the repository and install the required packages listed in the requirements file.

## Usage

### Data Loading and Visualization

The dataset is loaded, and images are visualized by centers. This step helps in understanding the distribution and variation in the dataset.

### Image Preprocessing

Images are cropped to a standard size to ensure uniformity. This preprocessing step is crucial for training the model effectively.

### Dataset Splitting

The dataset is split into training and validation sets. This split is done based on patient IDs to ensure that images from the same patient are not present in both training and validation sets, which helps in better generalization of the model.

### Dataloader Creation

Dataloaders are created for both training and validation sets. These dataloaders are used to efficiently load data in batches during model training and evaluation.

### Model Training

A pretrained Google LeNet model is fine-tuned on the dataset. The training process includes setting up the loss function, optimizer, and learning rate scheduler. The model is trained for a specified number of epochs, and the best-performing model is saved.

### Model Evaluation

The trained model is evaluated on a test dataset. The evaluation includes calculating the Area Under the Curve (AUC) for different OMERACT scores, which measure the model's performance in classifying images.

## Results

The models achieved the following AUC scores on the validation set:

- **General Model:**
  - OMERACT 0: 0.9115
  - OMERACT 1: 0.9423
  - OMERACT 2: 0.7604
  - OMERACT 3: 0.9667
  - Average: 0.8952

- **Submandibular Model:**
  - OMERACT 0: 0.9429
  - OMERACT 1: 1.0000
  - OMERACT 2: 1.0000
  - OMERACT 3: 1.0000
  - Average: 0.9857

- **Parotid Model:**
  - OMERACT 0: 0.8909
  - OMERACT 1: 1.0000
  - OMERACT 2: 0.6923
  - OMERACT 3: 1.0000
  - Average: 0.8958

## Authors:
  - Ángel de la Linde Valdés 
  - María Ordieres Álvarez
  - Alexia Durán Vizcaíno

## License

This project is licensed under the MIT License. See the license file for more details.
