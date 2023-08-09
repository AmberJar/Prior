# coding=utf-8
import sys

import ttach as tta
import os
from torchstat import stat

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

import argparse
import json
import models
from glob import glob

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
from collections import OrderedDict
import time
import torch.nn.functional as F
from tqdm import tqdm

tic = time.time()
window_size = 1024
from torch import nn
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# /data/chenyuxia/outputs/river_plant_pg/512/roads/img
def parse_arguments():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-c', '--config', default='config_loveda.json', type=str,
                        help='The config used to train the model')
    parser.add_argument('-m', '--model',
                        default='/data/fpc/saved/LoveDA/2023-07-28@21:34:15/checkpoint-epoch422.pth',
                        type=str,
                        help='Path to the .pth model checkpoint to be used in the prediction')
    parser.add_argument('-i', '--images', default=r'/data/fpc/data/love_DA/Test', type=str,
                        help='Path to the images to be segmented')
    parser.add_argument('-e', '--extension', default='png', type=str,
                        help='The extension of the images to be segmented')
    parser.add_argument("-t", "--tta", help="Test time augmentation.", default="d4", choices=[None, "d4", "lr"])
    args = parser.parse_args()
    return args


args = parse_arguments()
to_tensor = transforms.ToTensor()
num_classes = 7

# Model
print("Load model ............")
model = models.HiResNet(num_classes=num_classes, backbone='hrnet48')
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# Load checkpoint
print(args.model)
checkpoint = torch.load(args.model, map_location='cpu')
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
    checkpoint = checkpoint['state_dict']

# If during training, we used data parallel
if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
    # for gpu inference, use data parallel
    if "cuda" in device.type:
        model = torch.nn.DataParallel(model)
    else:
        # for cpu inference, remove module
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]
            new_state_dict[name] = v
        checkpoint = new_state_dict

# load

model.to(device)
model.load_state_dict(checkpoint)
model.eval()
print("Load model complete.>>>")


class SegmentationDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.transforms = transforms.Compose([
            transforms.Normalize(mean=[0.280082, 0.299398, 0.307035],
                                 std=[0.127366, 0.109451, 0.115518]),
        ])
        self.filenames = os.listdir(self.root)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.filenames[index])
        image = Image.open(image_path).convert('RGB')

        image = self.transforms(to_tensor(image))
        return image, self.filenames[index]

    def __len__(self):
        return len(self.filenames)


def predicts(image_path, model):
    if args.tta == "lr":
        tta_transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, tta_transforms)
    elif args.tta == "d4":
        tta_transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, tta_transforms)

    batch_size = 4
    testset = SegmentationDataset(root=image_path)
    print(len(testset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=64)
    tbar = tqdm(testloader, ncols=100)
    save_dir = '/data/fpc/data/love_DA/test_results'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        start = time.time()
        for image, name in tbar:
            preds = model(image.cuda())

            raw_predictions = torch.softmax(preds, dim=1).argmax(1).cpu().numpy()
            for i in range(batch_size):
                cv2.imwrite(os.path.join(save_dir, name[i]),
                           raw_predictions[i].astype(np.uint8))

        end = time.time() - start
        print(end)

predicts(args.images, model)
