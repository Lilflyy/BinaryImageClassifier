import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from skimage import color
from PIL import Image
import numpy as np

class customDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        image = io.imread(self.annotations.iloc[index, 0])
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)
        return (image, y_label)