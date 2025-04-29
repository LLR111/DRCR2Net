import numpy as np
from torch.utils.data import Dataset
import os
import torch
from PIL import Image

class TrainDataset(Dataset):
    def __init__(self, img_file, mask_file,transform=None):
        np.random.seed()
        self.transform = transform
        self.imgs_path = [img_file + '/' + path for path in sorted(os.listdir(img_file), key=lambda x: int(x.split('.')[0]))]
        self.labels_path = [mask_file + '/' + path for path in sorted(os.listdir(mask_file), key=lambda x: int(x.split('.')[0]))]

        indices = np.random.permutation(len(self.imgs_path))
        self.imgs_path = [self.imgs_path[i] for i in indices]
        self.labels_path = [self.labels_path[i] for i in indices]


    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        label_path = self.labels_path[idx]

        img = np.array(Image.open(img_path))/255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32, device='cuda')

        label = np.array(Image.open(label_path))/255.
        if len(label.shape) == 3:
            label = label[:,:,0]

        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(0)

        return img, label
