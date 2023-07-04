import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hiresnet_config import MODEL_CONFIGS
from timm.models.layers import trunc_normal_, DropPath
from blocks.CA import CA

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1, drop_path=0.):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes, momentum=bn_momentum)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(num_features=planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ca = CA(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = self.relu(out)

        out = self.ca(out)
        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.drop_path(out) + residual
        # out = self.relu(out)

        return out
