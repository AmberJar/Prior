from base import BaseDataSet, BaseDataLoader

from glob import glob
import numpy as np
import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from helper.mask_helper import grid_mask,block_mask,random_mask
ignore_label = 255

class CityScapesDataset(BaseDataSet):
    def __init__(self, mode, **kwargs):
        self.num_classes = 19
        self.mode = mode
        super(CityScapesDataset, self).__init__(**kwargs)

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



class CityScapes(DataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, mode='random_mask', val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [0.17613647, 0.18099176, 0.17772235]

        kwargs = {
            'root': data_dir,
            'mode': mode,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = CityScapesDataset(mode=mode, **kwargs)
        super(CityScapes, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)


