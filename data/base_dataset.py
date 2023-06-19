import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from .augmentor import transform_aug
import scipy.io as io


class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, random_aug=True):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.random_aug = random_aug
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.mean, self.std)

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError

    def _load_data(self, index):
        raise NotImplementedError

    def _augmentation(self, image, label):
        h, w, _ = image.shape

        # 随机增强！
        if self.random_aug:
            # see augmentor.py file and changed it for you own tasks or do more valdataions.
            transformed = transform_aug(height=h, width=w)(image=image, mask=label)
            image, label = transformed['image'], transformed['mask']

        return image, label

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)

        if self.split == 'train':
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))

        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

