import os
from PIL import Image
from torch_geometric.data import Dataset, Data
import torch
from torchvision import transforms
import random
import numpy as np

class ImageAugDataset(Dataset):
    def __init__(self, path):
        super(ImageAugDataset, self).__init__()
        self.img_path = os.path.join(path)

        self.train_files=os.listdir(self.img_path)

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.train_files[index])

        img = Image.open(img_name)

        img1 = self.random_aug(img)
        img2 = self.random_aug(img)

        return img1, img2

    def random_aug(self, img):
        RandomResizeCrop = transforms.RandomResizedCrop(224, scale=(0.2, 1))
        ColorJitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        RandomGrayscale = transforms.RandomGrayscale(p=0.2)
        RandomHorizontalFlip = transforms.RandomHorizontalFlip()
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize(
                                        mean=[0.1614, 0.1647, 0.1133], # l8 b3
                                        std=[0.1219, 0.0974, 0.0817] # l8 b3
                                         )

        img = RandomResizeCrop(img)
        if random.random() > 0.2:
            img = ColorJitter(img)
        img = RandomGrayscale(img)
        img = RandomHorizontalFlip(img)
        img = toTensor(img)
        img = normalize(img)
        return img
    
    def __len__(self):
        return len(self.train_files)
    def get(self):
        pass
    def len(self):
        pass
