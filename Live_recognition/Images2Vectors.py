from Siamese_model import SiameseNetwork
import torch
import os
import cv2
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np


def save_embeddings_to_file(embeddings, names, file):
    """
    Function that saves embeddings to a file.
    Args:
        embeddings: list of embeddings
        names: list of names
        file: path to file where embeddings will be saved
    """
    # Convert the embeddings to a numpy array
    embeddings_np = np.array([e.numpy() for e in embeddings])

    # Save the names to a text file
    with open(file + '_names.txt', 'w') as f:
        for name in names:
            f.write(f"{name}\n")

    # Save the embeddings to a text file
    np.savetxt(file + '_embeddings.txt', embeddings_np, fmt='%s')
    

def load_embeddings_from_file(file):
    """
    Function that loads embeddings from a file.
    Args:
        file: path to file where embeddings are saved
    """
    # Load the names from a text file
    with open(file + '_names.txt', 'r') as f:
        names = [line.strip() for line in f]

    # Load the embeddings from a text file
    embeddings = np.loadtxt(file + '_embeddings.txt')
    embeddings = torch.tensor(embeddings)

    return names, embeddings


def images2vectors(image_folder,csv_path, model):
    """
    Function that takes images and returns their embeddings.
    Args:
        images: list of images
        csv_path: list of image names
        model: SiameseNetwork model
    Returns:
        output: embeddings of images with shape (N, 4096)
                image names with shape (N,)

    """

    # Transform images
    transform = transforms.Compose([
            
            transforms.Resize((100, 100)),  # Resize images 
            # transforms.Grayscale(num_output_channels=3),  # Convert images to grayscale
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
    
    # read image names nad the label names
    df = pd.read_csv(csv_path, names=['Image', 'Name'],header=None)    
    images=[]
    for i in range(len(df)):
        images.append(Image.open(os.path.join(image_folder, df['Image'][i])))
    
    images = [transform(image) for image in images]
    tensor_images = torch.stack(images)  # Stack images into a single tensor
    with torch.no_grad():
        output = model.embedding(tensor_images)
    return output, df['Name'].values