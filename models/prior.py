import sys

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from itertools import chain
from collections import OrderedDict

from models.unet import UNet, UNetResnet
from models.hrnet import HRNet_W48_OCR
from torchvision import models as baseline


class PriorNet(nn.Module):
    def __init__(self, num_classes, **_):
        super(PriorNet, self).__init__()
        self.baseline = HRNet_W48_OCR(num_classes=num_classes, backbone="hrnet48")
        self.weight_path = '/data/fpc/projects/Prior/pretrained/hrnet_w48_ocr_1_latest.pth'
        self.load_model()
        self.backbone_freeze()

        self.num_classes = num_classes
        in_channels = 4  # image + mask
        self.unet = UNetResnet(num_classes, in_channels=in_channels)

    def forward(self, x):
        image = x[0]
        label = x[1]
        # first step 求backbone的结果
        bk_res, _ = self.baseline(image)

        # second step 这里对fusion的图片做处理
        fusion_features = torch.cat([image, label], dim=1)

        # third step 这里做压缩和重建
        features = self.unet(fusion_features)

        # fourth step 这里做和原来结果的融合
        # features = features + bk_res

        return features + bk_res

    def inference(self, x):
        image = x[0]

        # first step 求backbone的结果
        bk_res, _ = self.baseline(image)
        label = bk_res
        # second step 这里对fusion的图片做处理
        fusion_features = torch.cat([image, label], dim=1)

        # third step 这里做压缩和重建
        features = self.unet(fusion_features)

        # fourth step 这里做和原来结果的融合
        # features = features + bk_res

        return features + bk_res

    def load_model(self):
        print('start load model')
        checkpoint = torch.load(self.weight_path, map_location='cpu')

        checkpoint = checkpoint['state_dict']

        if 'module' in list(checkpoint.keys())[0]:
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict

        self.baseline.load_state_dict(checkpoint, strict=True)
        print('load success')

    def backbone_freeze(self):
        for name, parameters in self.baseline.named_parameters():
            # print(name, parameters.requires_grad)
            parameters.requires_grad = False


if __name__ == '__main__':
    model = PriorNet(num_classes=16, backbone='hrnet48')
    from torchinfo import summary

    summary(model, input_size=[(6, 3, 256, 256), (6, 3, 256, 256)])
    x_in = torch.rand(size=(6, 3, 256, 256)).cuda()
    outs = model(x_in)

    for out in outs:
        print(out.shape)