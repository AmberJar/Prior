import datetime
import time
import os
import cv2
import torch

path = '/data/fpc/data/Millon_AID/all_data/data/train_dir/train/images'

data_list = os.listdir(path)
print(len(data_list))

a = [x.item() for x in torch.linspace(0, 1e-4, 5)]

print(a)