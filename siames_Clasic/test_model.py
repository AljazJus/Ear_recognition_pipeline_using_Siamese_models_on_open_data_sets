import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import pandas as pd
import gc
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate(model, test_loader,foldername,device):
    model.eval()
    predictions = []
    labels = []
    progress_bar_test = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, batch in progress_bar_test:
        (image1, image2, label) = batch
        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device).view(-1, 1)
        output = model(image1, image2)
        # Convert model output to label
        predicted_labels = (output > 0.5).float()
        predictions.extend(predicted_labels.tolist())
        labels.extend(label.tolist())

    # Calculate precision and recall
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1= f1_score(labels, predictions)
    accuracy = accuracy_score(labels, predictions)

    # Save the evaluation metrics to test.jason file
    with open(os.path.join(foldername, 'test.json'), 'w') as file:
        json.dump({'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}, file)
  
def loss_graf(data,foldername):
    # Extract the training and validation loss
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    # Create a figure
    # Create a figure
    plt.figure()
    # Plot the training and validation loss
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    # Add a legend
    plt.legend()
    # Save the plot as an image file
    plt.savefig(os.path.join(foldername, 'loss.png'))
    # Close the figure to free up memory
    plt.close()

def predict(model, batch, foldername,device):
    model.eval()
    image1, image2, label = batch
    image1 = image1.to(device)
    image2 = image2.to(device)
    label = label.to(device)
    output = model(image1, image2)
    predicted_labels = (output > 0.5).float()

    # Draw the images and the labels
    fig, axs = plt.subplots(2,len(image1), figsize=(len(image1)*5,10 ))
    for i in range(len(image1)):
        axs[0, i].imshow(image1[i].permute(1, 2, 0))
        axs[1, i].imshow(image2[i].permute(1, 2, 0))
        axs[0, i].set_title(f"True: {label[i]}, Predicted: {predicted_labels[i]}")
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.tight_layout()
    # Save the figure to the specified folder
    plt.savefig(os.path.join(foldername, 'predictions.png'))
    # Close the figure to free up memory
    plt.close()
    
    


    