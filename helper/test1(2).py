import os
import sys
import random
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm
def random_mask(mask_path,mask_ratio,patch_size,save_path):
    mask = cv2.imread(mask_path,-1)
    h,w = mask.shape[0], mask.shape[1]
    num_patchs = (h//patch_size) * (w//patch_size)
    drop = int(num_patchs * mask_ratio)
    num_shuffle = np.random.randn(num_patchs)  # noise in [0, 1]
    # sort noise for each sample
    ids = np.argsort(num_shuffle)
    ids = ids[:drop]
    for index in ids:
        new_h = (index + 1) // (w//patch_size)
        new_w = (index + 1) % (w//patch_size) - 1
        if new_w == -1:
            new_w = (w//patch_size) -1
            new_h -= 1
        mask[new_h*patch_size:(new_h+1) * patch_size,new_w * patch_size:(new_w+1) * patch_size] = 255
    # cv2.imwrite(os.path.join(save_path,mask_path.split('/')[-1]),mask)
    return mask

def masks(mask,mask_path,min_num_patch,max_mask_patches,patch_size,save_path):
    delta = 0
    mask_ = cv2.imread(mask_path,-1)
    height, width = mask_.shape[0], mask_.shape[1]
    log_aspect_ratio = (math.log(0.3), math.log(1/0.3))
    for attempt in range(10):
        target_area = random.uniform(min_num_patch, max_mask_patches)
        aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w * patch_size < width and h * patch_size < height:
            top = random.randint(0, height//patch_size - h)
            left = random.randint(0, width//patch_size - w)

            num_masked = np.sum(mask[top*patch_size: top*patch_size + h * patch_size, 0: left*patch_size + w*patch_size] == 255)
            # Overlap
            if 0 < patch_size **2 *h *w - num_masked <= max_mask_patches * patch_size ** 2:
                for i in range(h):
                    for j in range(w):
                        if mask[(top+i)*patch_size:(top+i+1)*patch_size,(left+j)*patch_size:(left+j+1)*patch_size].all() != 255:
                            mask[(top + i) * patch_size:(top + i + 1) * patch_size,
                            (left + j) * patch_size:(left + j + 1) * patch_size] = 255
                            delta += 1
            if delta > 0:
                break

    return delta,mask

def block_mask(mask_path,save_path,masking_ratio,min_num_patch=4,patch_size=128):
    mask = cv2.imread(mask_path,-1)
    mask_count = 0
    h, w = mask.shape[0], mask.shape[1]
    num_masking_patches = (h // patch_size) * (w // patch_size) * masking_ratio
    while mask_count <= num_masking_patches:
        max_mask_patches = num_masking_patches - mask_count
        # max_mask_patches = min(max_mask_patches, max_num_patches)
        delta,mask = masks(mask,mask_path,min_num_patch,max_mask_patches,patch_size,save_path)

        if delta == 0:
            break
        else:
            mask_count += delta

    # cv2.imwrite(os.path.join(save_path, mask_path.split('/')[-1]), mask)
    return mask

def grid_mask(mask_path,save_path,patch_size=128):
    mask = cv2.imread(mask_path,-1)
    h,w = mask.shape[0], mask.shape[1]
    patch_h,patch_w = h//patch_size,w//patch_size
    for i in range(0,patch_h,2):
        for j in range(0,patch_w,2):
            mask[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size] = 255
    # cv2.imwrite(os.path.join(save_path, mask_path.split('/')[-1]), mask)
    return mask



def maskgenerator(train_path,val_path,save_train,save_val):
    for file in tqdm(os.listdir(train_path)):
        for labels in os.listdir(os.path.join(train_path,file)):
            if labels.split('.')[0].split('_')[-1] == 'labelTrainIds':
                label_path = os.path.join(train_path,file,labels)
                # random_mask(label_path,0.25,128,save_train)
                # block_mask(label_path,save_train,0.25,min_num_patch=4,patch_size=128)
                grid_mask(label_path,save_train,128)
    for file in tqdm(os.listdir(val_path)):
        for labels in os.listdir(os.path.join(val_path,file)):
            if labels.split('.')[0].split('_')[-1] == 'labelTrainIds':
                label_path = os.path.join(val_path,file,labels)
                # block_mask(label_path, save_val, 0.25, min_num_patch=4, patch_size=128)
                # random_mask(label_path,0.25,128,save_val)
                grid_mask(label_path, save_val, 128)




if __name__ == '__main__':

    maskgenerator(r'/data/chenyuxia/Cityspaces/gtFine/train',r'/data/chenyuxia/Cityspaces/gtFine/val',
                  None,None)


