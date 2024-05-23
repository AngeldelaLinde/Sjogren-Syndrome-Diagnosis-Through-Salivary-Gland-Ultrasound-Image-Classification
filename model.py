import os
import torch
import pandas as pd
from sklearn import metrics
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from PIL import Image

def computeAUCs(scores, labels):
    """
    Compute the Area Under the Curve (AUC) for each class.

    Args:
        scores (np.ndarray): Array of shape (n_samples, n_classes) with predicted scores for each class.
        labels (np.ndarray): Array of shape (n_samples,) with true class labels.

    Returns:
        np.ndarray: AUC values for each class.
    """
    num_classes = 4
    aucs = np.zeros((num_classes,))

    for i in range(num_classes):
        scores_class = scores[:, i]
        labels_class = (labels == i).astype(int)
        aucs[i] = metrics.roc_auc_score(labels_class, scores_class)

    return aucs

def train_model(dataloaders,dataset_sizes, model, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    numClasses = 4

    best_model_wts = copy.deepcopy(model.state_dict())
    best_aucs = np.zeros((numClasses,)) 
    best_auc = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set the model in training mode
            else:
                model.eval()   # Set the model in val mode (no grads)
            

            #Dataset size
            numSamples = dataset_sizes[phase]

            # Create variables to store outputs and labels
            outputs_m=np.zeros((numSamples,numClasses),dtype=float)
            labels_m=np.zeros((numSamples,),dtype=int)
            running_loss = 0.0

            contSamples=0

            # Iterate (loop of batches)
            for sample in dataloaders[phase]:
                inputs = sample['image'].to(device).float()
                labels = sample['label'].to(device)

                #Batch Sizea
                batchSize = labels.shape[0]

                # Set grads to zero
                optimizer.zero_grad()

                # Forward
                # Register ops only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward & parameters update only in train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Accumulate the running loss
                running_loss += loss.item() * inputs.size(0)

                #Apply a softmax to the output
                outputs=F.softmax(outputs.data,dim=1)
                # Store outputs and labels
                
                outputs_m [contSamples:contSamples+batchSize,...]=outputs.cpu().numpy()
                labels_m [contSamples:contSamples+batchSize]=labels.cpu().numpy()
                contSamples+=batchSize

            #At the end of an epoch, update the lr scheduler
            if phase == 'train':
                scheduler.step()

            
            epoch_loss = running_loss / dataset_sizes[phase]
            aucs=computeAUCs(outputs_m,labels_m)
            epoch_auc = aucs.mean()

            print(f"{phase} Loss: {epoch_loss:.4f} AUC OMERACT 0: {aucs[0]:.4f} OMERACT 1: {aucs[1]:.4f} OMERACT 2: {aucs[2]:.4f} OMERACT 3: {aucs[3]:.4f} avg: {epoch_auc:.4f}")

            # Deep copy of the best model
            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_aucs = aucs.copy()
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        print()

    time_elapsed = time.time() - since

    print(f"Best model in epoch {best_epoch:d} val AUC OMERACT 0: {best_aucs[0]:.4f} OMERACT 1: {best_aucs[1]:.4f} OMERACT 2: {best_aucs[2]:.4f} OMERACT 3: {best_aucs[3]:.4f} avg: {best_auc:.4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def createFTNetworkGoogLeNet():
    """
    Create and modify a pre-trained GoogLeNet model for fine-tuning.

    Returns:
        torch.nn.Module: Modified GoogLeNet model with the final fully connected layer adapted for 4 classes.
    """
    ftNet = models.googlenet(pretrained=True)
    
    num_ftrs = ftNet.fc.in_features
    num_classes = 4  
    ftNet.fc = nn.Linear(num_ftrs, num_classes)
    
    return ftNet

def test_model(model, test_dataset, test_dataloader, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): Trained model to be evaluated.
        test_dataset (torch.utils.data.Dataset): The test dataset.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to perform computations on (CPU or GPU).

    Returns:
        np.ndarray: Predicted scores for each class for all samples in the test dataset.
    """
    since = time.time()

    model.eval()   
    numSamples = len(test_dataset)
    numClasses = 4

    outputs_m = np.zeros((numSamples, numClasses), dtype=float)
    labels_m = np.zeros((numSamples,), dtype=int)
    contSamples = 0

    for sample in test_dataloader:
        inputs = sample['image'].to(device).float()

        batchSize = inputs.shape[0]

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            outputs = F.softmax(outputs.data, dim=1)
            outputs_m[contSamples:contSamples + batchSize, ...] = outputs.cpu().numpy()
            contSamples += batchSize

    return outputs_m
