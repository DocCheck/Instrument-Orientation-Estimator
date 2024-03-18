import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None, num_class=360):
        """
        img_list : list of images
        label_list : list of labels
        transform : the augmentation transforms list
        num_class : number of classes (angles) in degrees
        """
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform
        self.num_class = num_class

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.img_list[idx]
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        label = int(self.label_list[idx] / 1)  # angel intervals = 1

        if self.transform:
            image = self.transform(image)

        return image, label
