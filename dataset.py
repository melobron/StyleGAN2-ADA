import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import cv2

from utils import *


class FaceDataset(Dataset):
    def __init__(self, dataset='SEM1', train=True, transform=None):
        super(FaceDataset, self).__init__()

        data_dir = os.path.join('../datasets/', dataset)
        if train:
            self.img_dir = os.path.join(data_dir, 'train')
        else:
            self.img_dir = os.path.join(data_dir, 'test')
        self.img_paths = sorted(make_dataset(self.img_dir))

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB) / 255.
        img = self.transform(img)
        img = img.type(torch.FloatTensor)
        return img

    def __len__(self):
        return len(self.img_paths)
