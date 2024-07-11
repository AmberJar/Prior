import os
import time

import torch
import random
import numpy as np
from helper import HDDTBinaryLoss
from helper import HDLoss


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    seed_everything(0)
    pred = torch.rand(size=(6, 7, 256, 256)).cuda()
    target = torch.randint(low=0, high=6, size=(6, 256, 256)).cuda()
    hd_loss = HDDTBinaryLoss()
    hd_loss2 = HDLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)
    loss = hd_loss(pred, target)
    loss2 = hd_loss2(pred, target)

    start = time.time()
    for i in range(100):
        hd_loss = HDDTBinaryLoss()
    print('hd_cpu cost: ', time.time() - start)

    start = time.time()
    for i in range(100):
        hd_loss2 = HDLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)
    print('hd_gpu cost', time.time() - start)
    print(loss)
    print(loss2)
    # make something new