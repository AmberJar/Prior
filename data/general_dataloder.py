from glob import glob
import os
from torch.utils.data import Dataset, DataLoader
from helper.mask_helper import grid_mask,block_mask,random_mask
ignore_label = 255
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from .augmentor import transform_aug
from torch.utils.data.distributed import DistributedSampler



class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, mode, num_classes, random_aug=True):
        self.root = root
        self.num_classes = num_classes
        self.split = split
        self.mean = mean
        self.std = std
        self.random_aug = random_aug
        self.mode = mode
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.mean, self.std)

    def _set_files(self):
        self.image_dir = os.path.join(self.root, self.split, "images")
        self.label_dir = os.path.join(self.root, self.split, "labels")
        # print(self.image_dir, "\n", self.label_dir)
        self.files = [os.path.basename(path).split('.')[0] for path in
                      glob(self.image_dir + '/*.png')]

    def _load_data(self, index):
        image_id = self.files[index]
        label_id = image_id

        image_path = os.path.join(self.image_dir, image_id + '.png')
        label_path = os.path.join(self.label_dir, label_id + '.png')

        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)

        if self.mode == 'grid_mask':
            mask = grid_mask(label_path, None, 128)
        elif self.mode == 'block_mask':
            mask = block_mask(label_path, None, 0.25, 4, 128)
        else:
            mask = random_mask(label_path, 0.25, 125, None)
        return image, label, mask, image_id

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
        cv2.setNumThreads(0)
        image, label, mask, image_id = self._load_data(index)

        if self.split == 'train':
            image, label = self._augmentation(image, label)

        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        mask = torch.from_numpy(np.array(mask, dtype=np.int32)).long().unsqueeze(0)
        image = Image.fromarray(np.uint8(image))

        return self.normalize(self.to_tensor(image)), label, mask

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str


class General(DataLoader):
    def __init__(self, data_dir, batch_size, split, num_workers, num_classes, mode='random_mask', augment=False):

        self.MEAN = [0.286896, 0.283892, 0.325133]
        self.STD = [0.142422, 0.142357, 0.146488]

        kwargs = {
            'root': data_dir,
            'mode': mode,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'random_aug': augment,
            'num_classes': num_classes,
        }

        self.dataset = BaseDataSet(**kwargs)
        self.sampler = DistributedSampler(self.dataset, shuffle=True)
        super(General, self).__init__(self.dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=False,
                                         pin_memory=True,
                                         sampler=self.sampler,
                                         drop_last=True)
