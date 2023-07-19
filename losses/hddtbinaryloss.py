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


class HDDTBinaryLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super(HDDTBinaryLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        """
        preds: (batch_size, 2, x,y)
        target: ground truth, shape: (batch_size, x,y)
        """
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

    files = sorted(glob("*.tif"))
    images = []
    for f in files*100:
        img = cv2.imread(f, 0)
        img[img>0]=1
        img = np.expand_dims(img, axis=0)
        images.append(img)

    target = torch.from_numpy(np.concatenate(images, axis=0))
    print("target", target.shape)
    # =======================================================================================================
    print("test result is same or not.")
    t1 = time.time()
    res1 = compute_edts_forhdloss_thread(target)
    print("compute_edts_forhdloss_thread spend",time.time() - t1)

    t2 = time.time()
    res2 = compute_edts_forhdloss(target)
    print("compute_edts_forhdloss spend", time.time() - t1)
    print(np.sum(res2-res1))

    dist = np.concatenate((res1[0], res2[0], res1[1], res2[1]), axis=1)
    dist2 = np.concatenate((res1[2], res2[2], res1[3], res2[3]), axis=1)
    dist3 = np.concatenate((res1[4], res2[4], res1[5], res2[5]), axis=1)
    dist4 = np.concatenate((res1[6], res2[6], res1[7], res2[7]), axis=1)
    dist = np.concatenate([dist, dist2, dist3, dist4], axis=0)

    plt.imshow(dist)
    plt.show()

    # =======================================================================================================
    # 对比多线程和forloop花费时间
    # outs = torch.from_numpy(np.concatenate(images, axis=0))
    # outs = outs.unsqueeze(1)
    # outs = torch.cat([1-outs, outs], dim=1).type(torch.float32)
    # fun = HDDTBinaryLoss()
    # import time
    # t1 = time.time()
    # for i in range(10):
    #     b = fun(outs, target)
    # print(time.time() - t1)
    # print(b)
    # compute_edts_forhdloss_thread: 19.619222402572632
    # compute_edts_forhdloss:        38.723613262176514
    # =======================================================================================================

