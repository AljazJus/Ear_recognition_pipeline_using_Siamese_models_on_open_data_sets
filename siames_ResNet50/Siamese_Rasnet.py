# siamese.py

from torch.utils.data import Dataset
import torch.nn as nn
import torch
import random
import numpy as np
import os
from PIL import Image
    
class SiameseNetwork(nn.Module):
    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()
        self.embedding = base_model
        self.classifier = nn.Linear(2048, 1)

    def forward(self, input1, input2):
        output1 = self.embedding(input1)
        output2 = self.embedding(input2)
        l1_distance = torch.abs(output1 - output2)
        output = self.classifier(l1_distance)
        return output