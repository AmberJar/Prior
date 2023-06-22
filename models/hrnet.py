import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hrnet_spatial_ocr_block import SpatialOCR_ASP_Module
from models.hrnet_backbone import HRNetBackbone
from itertools import chain

import sys


class HRNet_W48_OCR(nn.Module):
    def __init__(self, num_classes, backbone, weight_path=None, freeze_bn=False, freeze_backbone=False, **_):
        super(HRNet_W48_OCR, self).__init__()
        self.num_classes = num_classes
        self.backbone = HRNetBackbone(backbone)

        # extra added layers
        in_channels = 720  # 48 + 96 + 192 + 384
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.SyncBatchNorm(512),
            nn.ReLU(inplace=True))

        from models.hrnet_spatial_ocr_block import SpatialGather_Module
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        from models.hrnet_spatial_ocr_block import SpatialOCR_Module
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.SyncBatchNorm(in_channels),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return out_aux, out

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.asp_ocr_head.parameters(), self.cls_head.parameters(), self.aux_head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


if __name__ == '__main__':
    model = HRNet_W48_ASPOCR(num_classes=16, backbone='hrnet48')
    from torchinfo import summary

    summary(model, input_size=(6, 3, 256, 256))
    x_in = torch.rand(size=(6, 3, 256, 256)).cuda()
    outs = model(x_in)

    for out in outs:
        print(out.shape)
