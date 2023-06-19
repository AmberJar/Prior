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


class CityScapesDataset(Dataset):
    def __init__(self, root, split, mean, std, mode, num_classes, random_aug=False):
        super(CityScapesDataset, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.random_aug = random_aug
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.mean, self.std)

    def _set_files(self):
        assert (self.mode in ['grid_mask','block_mask','random_mask'] and self.split in ['train', 'val'])

        SUFIX = '_gtFine_labelTrainIds.png'

        img_dir_name = 'leftImg8bit'
        label_dir_name = 'gtFine'
        label_path = os.path.join(self.root, label_dir_name, self.split)
        image_path = os.path.join(self.root, img_dir_name, self.split)
        assert os.listdir(image_path) == os.listdir(label_path)

        image_paths, label_paths = [], []
        for city in os.listdir(image_path):
            image_paths.extend(sorted(glob(os.path.join(image_path, city, '*.png'))))
            label_paths.extend(sorted(glob(os.path.join(label_path, city, f'*{SUFIX}'))))
        self.files = list(zip(image_paths, label_paths))

    def _load_data(self, index):
        image_path, label_path = self.files[index]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        if self.mode == 'grid_mask':
            mask = grid_mask(label_path,None,128)
        elif self.mode == 'block_mask':
            mask = block_mask(label_path,None,0.25,4,128)
        else:
            mask = random_mask(label_path,0.25,125,None)
        return image, label, mask, image_id

    def _augmentation(self, image, label):
        h, w, _ = image.shape

        # 随机增强！
        if self.random_aug:
            print('Aug!!!')
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


class CityScapes(DataLoader):
    def __init__(self, data_dir, batch_size, split, num_workers, num_classes, mode='random_mask', augment=False):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

        kwargs = {
            'root': data_dir,
            'mode': mode,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'random_aug': augment,
            'num_classes': num_classes,
        }

        self.dataset = CityScapesDataset(**kwargs)
        self.sampler = DistributedSampler(self.dataset, shuffle=True)
        super(CityScapes, self).__init__(self.dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         shuffle=False,
                                         pin_memory=True,
                                         sampler=self.sampler,
                                         drop_last=True)


