import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split

import torchvision.models as models
import os
from tqdm import tqdm
import json
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_path = '/home/aljazjustin/compear_model/model4'
data_dir = '/home/aljazjustin/datasets/NEW copy/final_data/train'
batch_size=64
num_epochs = 100

print(f'Device: {device}')
# Load the pretrained model
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
print(model)

# Check if multiple GPUs are available and wrap the model


# Modify the last layer
num_ftrs = model.classifier[1].in_features
num_classes = 1310  # Replace with the number of people you want to identify
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True)
)


if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    model = nn.DataParallel(model)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Load your data
# Replace with the path to your dataset
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load your data

dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Define the split size. For example, let's use 80% for training and 20% for validation.
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Split the data
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}

train_loss, val_loss = [], []
train_acc, val_acc = [], []

best_val_loss = float('inf')
patience = 0

# Train the model

torch.save(model.module.state_dict(), os.path.join(save_path, f'TEST1.pth'))

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        print(f'Epoch {epoch}/{num_epochs - 1}, Phase: {phase}')
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloaders[phase]):

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset if phase == 'train' else val_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset if phase == 'train' else val_dataset)

        # Append the loss and accuracy to the lists
        if phase == 'train':
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
        else:
            val_loss.append(epoch_loss)
            val_acc.append(epoch_acc)

        # Update the best validation loss and reset the patience counter
        if phase == 'val' and epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            patience = 0
        # If the validation loss did not improve, increment the patience counter
        elif phase == 'val':
            patience += 1

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Save the model
    if (epoch) % 5 == 0:
        torch.save(model.module.state_dict(), os.path.join(save_path, f'model_{epoch}.pth'))
    
    # If the validation loss did not improve for 5 epochs, stop training
    if patience == 5:
        break

torch.save(model.module.state_dict(), os.path.join(save_path,f'final_model.pth'))
# Your data
data = {'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc}

# Write to a JSON file
with open('data.json', 'w') as f:
    json.dump(data, f)

