import os
import os.path as osp
import numpy as np
import torch
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image
import random
from .augmentor import transform_aug
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from .augmentor import transform_aug
from torch.utils.data.distributed import DistributedSampler

CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]

ORIGIN_IMG_SIZE = (512, 512)


# INPUT_IMG_SIZE = (1024, 1024)
# TEST_IMG_SIZE = (1024, 1024)


def get_training_transform(mean, std):
    train_transform = [
        # albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.15),
        # albu.RandomRotate90(p=0.25),
        albu.Normalize(mean=mean, std=std)
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask, mean, std):
    height, width = img.size
    # crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
    #
    #                    SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=7, nopad=False)])
    # img, mask = crop_aug(img, mask)

    img, mask = np.array(img), np.array(mask)
    # aug = get_training_transform(mean, std)(image=img, mask=mask)
    aug = transform_aug(height, width, mean, std)(image=img, mask=mask)
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform(mean, std):
    val_transform = [
        albu.Normalize(mean=mean, std=std)
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask, mean, std):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform(mean, std)(image=img, mask=mask)
    img, mask = aug['image'], aug['mask']
    return img, mask


class HirResNetDataset(Dataset):
    def __init__(self, root, mean, std, num_classes, mosaic_ratio, split='val',
                 img_size=ORIGIN_IMG_SIZE):
        self.root = root
        self.img_suffix = '.tif'
        self.mask_suffix = '.png'

        self.mode = split
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.img_dir = os.path.join(root, self.mode, 'images')
        self.mask_dir = os.path.join(root, self.mode, 'labels')
        self.img_ids = self.get_img_ids(self.root, self.img_dir, self.mask_dir)
        self.mean = mean
        self.std = std
        self.num_classes = num_classes

        if self.mode == 'train':
            self.transform = train_aug
        else:
            self.transform = val_aug

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio >= self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask = self.load_img_and_mask(index)
            if self.transform:
                img, mask = self.transform(img, mask, self.mean, self.std)
        else:
            img, mask = self.load_mosaic_img_and_mask(index)
            img, mask = np.array(img), np.array(mask)
            aug = get_val_transform(self.mean, self.std)(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
            # if self.transform:
            #     img, mask = self.transform(img, mask, self.mean, self.std)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        img_id = self.img_ids[index]
        # results = dict(img_id=img_id, img=img, gt_semantic_seg=mask)
        return img, mask

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.root, self.mode, "images", img_id + self.img_suffix)
        mask_name = osp.join(self.root, self.mode, "labels", img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        return img, mask

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a, mask=mask_a)
        croped_b = random_crop_b(image=img_b, mask=mask_b)
        croped_c = random_crop_c(image=img_c, mask=mask_c)
        croped_d = random_crop_d(image=img_d, mask=mask_d)

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        return img, mask


class HiResNetDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, split, num_workers, num_classes, mosaic_ratio=0.25, mode='random_mask', augment=False):
        # Vaihingen
        # self.MEAN = [0.463633, 0.316652, 0.320528]
        # self.STD = [0.203334, 0.135546, 0.140651]

        # Potsdam
        self.MEAN = [0.337606, 0.333821, 0.360477]
        self.STD = [0.118292, 0.120414, 0.116505]
        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'num_classes': num_classes,
            'mosaic_ratio': mosaic_ratio,
        }

        self.dataset = HirResNetDataset(**kwargs)
        if split != 'test':
            self.sampler = DistributedSampler(self.dataset, shuffle=True)
        else:
            self.sampler = None

        super(HiResNetDataLoader, self).__init__(self.dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=False,
                                                 pin_memory=False,
                                                 sampler=self.sampler,
                                                 drop_last=True)
