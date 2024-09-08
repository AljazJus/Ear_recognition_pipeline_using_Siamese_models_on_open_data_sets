import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas as pd
import gc
import gpustat
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR
from pathlib import Path
from efficientnet_pytorch import EfficientNet

from test_model import evaluate, loss_graf, predict
from siamese import SiameseNetwork
from siamese_dataset2 import SiameseDataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

save_path="/home/aljazjustin/compear_model/siames_model1"
basem_model_path="/home/aljazjustin/compear_model/model4/final_model.pth"

train_files="/home/aljazjustin/datasets/NEW/final_data/train"
val_files="/home/aljazjustin/datasets/NEW/final_data/val"
test_files="/home/aljazjustin/datasets/NEW/final_data/test"

if not os.path.exists(save_path):
    # If not, create the directory
    print(f"Creating directory: {save_path}")
    os.makedirs(save_path)
else:
    print(f"Directory {save_path} already exists")
    save_path=save_path+"1"
    print(f"Creating directory: {save_path}")
    os.makedirs(save_path)

num_epochs = 100
save_checkpoint_dir = save_path+"/checkpoints"
batch_size=32*2
lr=2e-4
seed=42
torch.manual_seed(seed) 
np.random.seed(seed)
random.seed(seed)


data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_val = transforms.Compose([
    
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


if not os.path.exists(save_checkpoint_dir):
    # If not, create the directory
    print(f"Creating directory: {save_checkpoint_dir}")
    os.makedirs(save_checkpoint_dir)

# Make sure the checkpoint directory exists
def save_checkpoint(base_model,linear, optimizer,epo=0,name=None,checkpoint_dir = 'checkpoints'):
    checkpoint_prefix = os.path.join(save_path,checkpoint_dir, f'ckpt_{epo:03}.pth')
    if name is not None:
        checkpoint_prefix = os.path.join(save_path,checkpoint_dir, f'{name}.pth')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Save the state dictionaries of the model and optimizer
    torch.save({
        'model_state_dict': base_model.state_dict(),
        'linear_state_dict': linear.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_prefix)
    # print(f"Saving checkpoint: {checkpoint_prefix}")

#################################
# Load the data


num_workers = 8
# Define your own image paths and labels
dataset_train = SiameseDataset(train_files, transform=data_transforms)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

dataset_validation = SiameseDataset(val_files, transform=data_transforms_val)
validation_loader = DataLoader(dataset_validation, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

dataset_test = SiameseDataset(test_files, transform=data_transforms)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


####################################################################################################
# Define the device
# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Move the model to the GPU
if torch.cuda.is_available():
    print('Moving model to GPU')
    pid = os.getpid()
    print(f"Current PID is: {pid}")
else:
    print('Moving model to CPU')
    exit()

# If you're using CUDA, you can also empty the CUDA cache
gc.collect()
if device.type == 'cuda':
    torch.cuda.empty_cache()

############################################################
# Create the model, loss function, and optimizer
model_base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
# Modify the last layer
num_ftrs = model_base.classifier[1].in_features
num_classes = 1310  # Replace with the number of people you want to identify
model_base.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
)
state_dict = torch.load(basem_model_path)

model_base.load_state_dict(state_dict)
model_base.eval()

model_base.classifier = nn.Identity()
# Freeze the base model
for param in model_base.parameters():
    param.requires_grad = False

model= SiameseNetwork(model_base)
model = model.to(device)



# If there are multiple GPUs, wrap the model with nn.DataParallel
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)



criterion = nn.BCEWithLogitsLoss()
# criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005 , momentum=0.9)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)



# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
# ...
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# Move the loss function to the GPU
criterion = criterion.to(device)


# Define the directory for saving the checkpoints

os.makedirs(save_checkpoint_dir, exist_ok=True)


####################################################################################################
def train_step( batch):
    (input1, input2, labels)=batch
    input1 = input1.to(device)
    input2 = input2.to(device)
    labels = labels.to(device).view(-1, 1)
    # Forward pass
    prediction = model(input1, input2)
    # Calculate loss
    loss = criterion(prediction, labels)
    # Zero the gradients
    optimizer.zero_grad()
    # Calculate gradients
    loss.backward()
    # Update weights
    optimizer.step()
    # Return loss
    return loss.item()


def val_step(batch):
    (input1, input2, labels)=batch
    input1 = input1.to(device)
    input2 = input2.to(device)
    labels = labels.to(device).unsqueeze(1)
    # Forward pass
    prediction = model(input1, input2)

    # Calculate loss
    loss = criterion(prediction, labels)

    # Calculate accuracy
    prediction = (prediction > 0.5).float()
    accuracy = (prediction == labels).float().mean()

    # Return loss and accuracy
    return loss.item(), accuracy.item()

####################################################################################################
best_val_loss = float('inf')
best_accuracy = 0
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
}
patience = 5  # Number of epochs with no improvement after which training will be stopped.
patience_counter = 0

# Use the function in your training loop
for epoch in range(num_epochs+1):
    print(f"Epoch: {epoch}")
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    train_loss = 0
    model.train()
    for i,batch in progress_bar:
        loss=train_step(batch)
        progress_bar.set_description(f"Loss: {loss:.4f}")
        train_loss += loss

    scheduler.step()

    train_loss /= len(train_loader)
    history['train_loss'].append(train_loss)
    # print(f"Training loss: {train_loss:.4f}")
    # Save the model checkpoint
    
    
    progress_bar_val = tqdm(enumerate(validation_loader), total=len(validation_loader))
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Do not calculate gradients to save memory
        validation_loss = 0
        validation_accuracy = 0
        for i, batch in progress_bar_val:
            loss,acc=val_step( batch)
            progress_bar_val.set_description(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
            validation_loss += loss
            validation_accuracy += acc
        validation_accuracy /= len(validation_loader)
        validation_loss /= len(validation_loader)
        # scheduler.step(validation_loss)
    
    history['val_loss'].append(validation_loss)
    history['val_accuracy'].append(validation_accuracy)

    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        save_checkpoint(model.module.base_model, model.module.classifier, optimizer, name='best_model_acc')
        patience_counter = 0

    if validation_loss < best_val_loss:
        best_val_loss = validation_loss
        save_checkpoint(model.module.base_model, model.module.classifier, optimizer, name='best_model_loss')
        patience_counter = 0
    else:
        if epoch > 5:
            patience_counter += 1

    with open(os.path.join(save_path, 'history.json'), 'w') as f:
            json.dump(history, f)
    
    if epoch % 10 == 0:
        save_checkpoint(model.module.base_model, model.module.classifier, optimizer, epoch)
        
    torch.cuda.empty_cache()
    if patience_counter >= patience:
        print("Early stopping")
        break
    
    print(f"Epoch number: {epoch} \t Training loss: {train_loss:.4f} \t Validation loss: {validation_loss:.4f}\t Validation acc:{validation_accuracy:.4f}\n")

with open(os.path.join(save_path, 'history.json'), 'w') as f:
    json.dump(history, f)

save_checkpoint(model.module.base_model, model.module.classifier, optimizer, name='final_model')

loss_graf(history,save_path)

evaluate(model, test_loader,save_path,device)
# Save the final model

print('Finished Training')