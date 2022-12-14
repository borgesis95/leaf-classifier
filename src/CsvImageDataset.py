from torch.utils import data
from PIL import Image
import numpy as np
import pandas as pd

class CSVImageDataset(data.Dataset):
    def __init__(self,csv, transform = None):
        self.data = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        im_path, im_label = self.data.iloc[i]['image'], self.data.iloc[i].labels
        im = Image.open(im_path)
 
        if self.transform is not None:
            im = self.transform(im)
 
        return im, im_label

