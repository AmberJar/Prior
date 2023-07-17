"""
compute haudorff loss for binary segmentation
https://arxiv.org/pdf/1904.10030v1.pdf
"""

import torch
from torch import nn
from scipy.ndimage import distance_transform_edt
import numpy as np
import threading
import warnings
warnings.filterwarnings("ignore")


def get_single_edt(idx, segmentation, segmentation_outs):
    pos = segmentation[idx]
    neg = 1 - pos
    dst_pos = distance_transform_edt(pos)
    dst_neg = distance_transform_edt(neg)
    dst = dst_neg + dst_pos
    segmentation_outs[idx] = dst
    return


def compute_edts_forhdloss_thread(segmentation):
    segmentation_outs = np.zeros_like(segmentation)*1.
    idxs = [i for i in range(segmentation.shape[0])]
    threads = []
    for idx in idxs:
        # 实例化n个对象，target=目标函数名，args=目标函数参数(元组格式)
        t = threading.Thread(target=get_single_edt, args=(idx,segmentation,segmentation_outs,))
        threads.append(t)
        t.start()
    # 等待所有子线程结束再运行主线程
    [thread.join() for thread in threads]
    return segmentation_outs

# def get_distance_transform_weight(segmentation):
#     pos = [ele for ele in segmentation]
#     neg = [ele for ele in 1-segmentation]
#     array_pos = np.stack(map(distance_transform_edt, pos))
#     array_neg = np.stack(map(distance_transform_edt, neg))
#     dis = array_pos + array_neg
#     del pos, neg, array_pos, array_neg
#     gc.collect()
#     return dis


def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = 1-posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res


class EdgeAwareLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(EdgeAwareLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        """
        preds: (batch_size, n, x, y)
        target: ground truth, shape: (batch_size, x,y)
        """
        # 忽略掉不需要的值
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()

        gt = target.type(torch.float32)

        pc = torch.softmax(preds, dim=1)
        pc = torch.argmax(pc, dim=1) * 1.

        with torch.no_grad():
            # pc_dist = compute_edts_forhdloss(pc.cpu().numpy())
            # gt_dist = compute_edts_forhdloss(gt.cpu().numpy())
            pc_dist = compute_edts_forhdloss_thread(pc.cpu().numpy()) * 1.
            gt_dist = compute_edts_forhdloss_thread(gt.cpu().numpy()) * 1.

        pred_error = ((gt - pc) ** 2)**.5
        dist = (pc_dist ** 2 + gt_dist ** 2)**.5

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)
        multipled = torch.pow(torch.einsum("bxy,bxy->bxy", pred_error, dist) + 1e-9, .5)
        hd_loss = multipled.mean()

        return hd_loss


if __name__ == '__main__':
    import cv2
    from matplotlib import pyplot as plt
    from glob import glob
    import time

    """
    preds: (batch_size, n, x, y)
    target: ground truth, shape: (batch_size, x,y)
    """

    preds = torch.zeros((16, 6, 256, 256))
    target = torch.randint(0, 6)
