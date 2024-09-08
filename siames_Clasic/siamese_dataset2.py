

from torch.utils.data import Dataset
import torch.nn as nn
import torch
import random
import numpy as np
import os
from PIL import Image


def get_all_images(dir_path, extensions=['.jpg', '.png', '.jpeg']):
    images = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                images.append(os.path.join(root, file))

    return images


def get_all_folders(dir_path):
    return [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]

class SiameseDataset(Dataset):
    def __init__(self,images, transform=None):
        self.image_folder = images
        self.transform = transform
        self.image_paths = get_all_images(images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image1_name = self.image_paths[idx]
        image1_path= os.path.join(self.image_folder, image1_name)
        image1 = Image.open(image1_path)

        # Get a positive or negative pair
        if np.random.rand() > 0.5:  # Same class
            label = 1.0
            folder = os.path.dirname(image1_name)
            folder_files = [file for file in self.image_paths if folder in file]
            if len(folder_files) == 1:
                label = 0.0
                for i in range(100):
                    image2_name = random.choice(self.image_paths)
                    if os.path.dirname(image2_name) != os.path.dirname(image1_name):
                        break
            elif len(folder_files) == 2:
                image2_name = folder_files[0]
                if image1_name == folder_files[0]:
                    image2_name = folder_files[1]
            else:
                for i in range(100):
                    # print('ERROR No other image in the folder')
                    image2_name = random.choice(folder_files)
                    if os.path.dirname(image2_name) == os.path.dirname(image1_name):
                        if image1_name != image2_name:
                            break
        else:  # Different class
            for i in range(100):
                image2_name = random.choice(self.image_paths)
                if os.path.dirname(image2_name) != os.path.dirname(image1_name):
                    break
            else:
                # print('ERROR')
                image2_name = random.choice(self.image_paths)
            label = 0.0
        image2_path= os.path.join(self.image_folder, image2_name)
        image2 = Image.open((image2_path))

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(label, dtype=torch.float32)
