import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import torch


class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0,
                 warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmUpLR_CosineAnnealing(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, num_epochs, iters_per_epoch, init_lr=1e-7, depths=5, warmup_epochs=0, last_epoch=-1):
        # print(warmup_epochs)
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.warmup_epochs = warmup_epochs
        self.cur_iter = 0
        self.num_epochs = num_epochs
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.init_lr = init_lr
        self.depths = depths

        super(WarmUpLR_CosineAnnealing, self).__init__(optimizer, last_epoch)

    def _get_lr_list(self):
        warmup_list = [x.item() for x in torch.linspace(self.init_lr, self.base_lrs[0], self.depths)]
        return warmup_list

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        warmup_list = self._get_lr_list()
        # iter = epoch * iters_per_epoch + i
        # maxiter = num_epochs * iters_per_epoch
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor = (1 + math.cos(1.0 * T / self.N * math.pi))
        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1

        if self.last_epoch < self.warmup_epochs:
            return [warmup_list[self.last_epoch]]
            # return [base_lr * self.last_epoch / (self.warmup_iters + 1e-8) for base_lr in self.base_lrs]
        return [0.5 * base_lr * factor for base_lr in self.base_lrs]