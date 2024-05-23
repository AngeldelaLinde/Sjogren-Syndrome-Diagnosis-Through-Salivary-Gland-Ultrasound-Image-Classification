import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def create_dataloader(images, labels, batch_size=8, dataloader_type="test", target_size = (365, 496)):

    """
    Create a PyTorch DataLoader for training or testing with specified transformations.

    Args:
    - images (list): List of images.
    - labels (list): List of labels corresponding to the images.
    - batch_size (int): Batch size for the DataLoader (default is 32).
    - dataloader_type (str): Type of DataLoader, either "train" or "test" (default is "test").
    - target_size (tuple): Target size for image resizing (default is (365, 496)).

    Returns:
    - dataset (Dataset): PyTorch Dataset object.
    - dataloader (DataLoader): PyTorch DataLoader object.

    This function creates a PyTorch DataLoader for either training or testing purposes.
    It applies different transformations based on the type of DataLoader.
    For training DataLoader, it applies random horizontal flip, rotation with crop, random resized crop,
    Gaussian noise, and image normalization transformations.
    For testing DataLoader, it applies only image normalization transformation.
    """
    
    class GaussianNoise(object):
        def __init__(self, mean=0.0, std=0.1):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            return tensor + torch.randn(tensor.size()) * self.std + self.mean

    class RotateAndCrop(object):
        def __init__(self, degrees, target_size):
            self.degrees = degrees
            self.target_size = target_size

        def __call__(self, img):
            angle = transforms.RandomRotation.get_params([-self.degrees, self.degrees])
            img = transforms.functional.rotate(img, angle)
            # Aggressively crop the image to remove black corners
            width, height = img.size
            new_width = int(width * 0.8)
            new_height = int(height * 0.8)
            img = transforms.functional.center_crop(img, (new_height, new_width))
            img = transforms.functional.resize(img, self.target_size)
            return img


    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.25),  # Apply horizontal flip with 25% probability
        transforms.RandomApply([RotateAndCrop(degrees=30, target_size=target_size)], p=0.25),  # Apply rotation with crop to remove black corners
        transforms.RandomApply([transforms.RandomResizedCrop(size=target_size, scale=(0.8, 1.0))], p=0.25),  # Random crop with 25% probability
        transforms.ToTensor(),
        transforms.RandomApply([GaussianNoise(mean=0.0, std=0.1)], p=0.25),  # Apply Gaussian noise with 25% probability
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #Image normalization
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) #Image normalization
    ])

    class USDataset(Dataset):
        def __init__(self, images, labels=None, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            label = self.labels[idx] if self.labels is not None else None
            
            if self.transform:
                # Convert NumPy array to PIL Image
                image = Image.fromarray(image)
                # Apply transformations
                image = self.transform(image)
            
            sample = {'image': image, 'label': label}
            return sample
    
    if dataloader_type=="train":
        dataset = USDataset(images, labels, transform=train_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif dataloader_type=="val":
        dataset = USDataset(images, labels, transform=test_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif dataloader_type=="test":
        dataset = USDataset(images, labels, transform=test_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataset, dataloader

    

