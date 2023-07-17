import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hiresnet_spatial_ocr_block import BNReLU, SpatialOCR_ASP_Module
from models.hiresnet_backbone import HRNetBackbone
from itertools import chain


class HiResNetWithoutOCR(nn.Module):
    def __init__(self, num_classes, backbone, use_pretrained_backbone=False, pretrained_path=None, **_):
        super(HiResNetWithoutOCR, self).__init__()
        self.backbone = HRNetBackbone(backbone)
        self.num_classes = num_classes

        # extra added layers
        in_channels = 336  # 48 + 96 + 192 + 384
        # 64 + 128 + 256 + 512

        if use_pretrained_backbone:
            # print('success load pretrained weights')
            self._load_pretrained_model(_pretrained_weights=pretrained_path)

        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            BNReLU(512, bn_type=None),  # None
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()
        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        # feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=Tre)
        feats = torch.cat([feat1, feat2, feat3], 1)
        # coarse segmentation
        out_aux = self.aux_head(feats)
        out_aux = F.interpolate(out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        return out_aux

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.asp_ocr_head.parameters(), self.cls_head.parameters(), self.aux_head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

    def _load_pretrained_model(self, _pretrained_weights):
        checkpoint = torch.load(_pretrained_weights, map_location=torch.device('cpu'))
        state_dict = self.state_dict()
        model_dict = {}

        for k, v in checkpoint['state_dict'].items():
            model_dict[k] = v

        state_dict.update(model_dict)
        print('sucess')
        self.backbone.load_state_dict(state_dict, strict=False)
