import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils import class_weight
# from Hausdroff_Loss import HDDTBinaryLoss
from helper.hddtbinaryloss import HDDTBinaryLoss
import math
from helper import HDLoss


#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``:::::::fpc:::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2],
                                labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index,
                                      reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """
    def __init__(self, normalization='sigmoid', epsilon=1e-6, ignore_index=255):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.normalization = normalization
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=prediction.size()[1])
        output = F.softmax(prediction, dim=1)
        assert prediction.size() == target.size(), "'prediction' and 'target' must have the same shape"
        if prediction.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            prediction = torch.cat((prediction, 1 - prediction), dim=0)
            target = torch.cat((target, 1 - target), dim=0)
        prediction = torch.transpose(output, 1, 0)
        prediction = torch.flatten(prediction, 1) #flatten all dimensions except channel/class
        target = torch.transpose(target, 1, 0)
        target = torch.flatten(target, 1)
        target = target.float()
        w_l = target.sum(-1)
        w_l = 1 / (w_l ** 2).clamp(min=self.epsilon)
        w_l.requires_grad = False
        intersect = (prediction * target).sum(-1)
        intersect = intersect * w_l

        denominator = (prediction + target).sum(-1)
        # print(denominator)
        denominator = (denominator * w_l).clamp(min=self.epsilon)
        # print(denominator)
        return 1 - (2 * (intersect.sum() / denominator.sum()))


class LabelSmoothSoftmaxCE(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=255):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        # print(np.unique(label.cpu().numpy()))
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class LSCE_GDLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(LSCE_GDLoss, self).__init__()
        self.gdice = GeneralizedDiceLoss(ignore_index=ignore_index)
        self.LSCE = LabelSmoothSoftmaxCE(ignore_index=ignore_index)
        self.hdloss = HDLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)

    def forward(self, output, target):
        LSCE_loss = self.LSCE(output, target)
        gdice_loss = self.gdice(output, target)
        hd_loss = self.hdloss(output, target)
        # print('LSCE', LSCE_loss)
        # print('gdice_loss', gdice_loss)
        # print('hd_loss', hd_loss)

        return LSCE_loss + gdice_loss + 0.4 * hd_loss
        # return [LSCE_loss, gdice_loss, hd_loss]


class LSCE_GDLoss_GLW(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255,
                 weight=None):
        super(LSCE_GDLoss_GLW, self).__init__()
        self.gdice = GeneralizedDiceLoss(ignore_index=ignore_index)
        self.LSCE = LabelSmoothSoftmaxCE(ignore_index=ignore_index)
        self.hdloss = HDLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True)

    def forward(self, output, target):
        LSCE_loss = self.LSCE(output, target)
        gdice_loss = self.gdice(output, target)
        hd_loss = self.hdloss(output, target)
        # print('LSCE', LSCE_loss)
        # print('gdice_loss', gdice_loss)
        # print('hd_loss', hd_loss)

        # return LSCE_loss + gdice_loss + 0.4 * hd_loss
        return [LSCE_loss, gdice_loss, hd_loss]