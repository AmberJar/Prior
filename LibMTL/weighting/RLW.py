import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class RLW(AbsWeighting):
    r"""Random Loss Weighting (RLW).
    
    This method is proposed in `Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (TMLR 2022) <https://openreview.net/forum?id=jjtFD8A1Wx>`_ \
    and implemented by us.

    """
    def __init__(self, device, scaler, task_num=3):
        super(RLW, self).__init__()
        self.task_num = task_num
        self.device = device
        self.scaler = scaler

    def backward(self, losses, **kwargs):
        batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        losses = torch.tensor(losses).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        # loss = torch.nan_to_num(loss, nan=1e-8)
        loss.requires_grad_(True)

        # self.scaler.scale(loss).backward()
        return loss, batch_weight.detach().cpu().numpy()


class _RLW(AbsWeighting):
    r"""Random Loss Weighting (RLW).

    This method is proposed in `Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (TMLR 2022) <https://openreview.net/forum?id=jjtFD8A1Wx>`_ \
    and implemented by us.

    """

    def __init__(self):
        super(_RLW, self).__init__()

    def backward(self, losses, **kwargs):
        batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()
