import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hiresnet_spatial_ocr_block import BNReLU, SpatialOCR_ASP_Module
from models.hiresnet_backbone import HRNetBackbone
from itertools import chain
from collections import OrderedDict


class HiResNet(nn.Module):
    def __init__(self, num_classes, backbone, freeze_bn=False, use_pretrained_backbone=False, **_):
        super(HiResNet, self).__init__()
        self.backbone = HRNetBackbone(backbone)
        self.num_classes = num_classes

        # extra added layers
        # 336
        in_channels = 336  # 48 + 96 + 192 + 384
        # 64 + 128 + 256 + 512

        self.asp_ocr_head = SpatialOCR_ASP_Module(features=in_channels,
                                                  hidden_features=256,
                                                  out_features=256,
                                                  dilations=(24, 48, 72),
                                                  num_classes=self.num_classes,
                                                  bn_type=None)  # None

        self.cls_head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)

        if use_pretrained_backbone:
            self._load_checkpoint(use_pretrained_backbone)

        self.aux_head = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            BNReLU(512, bn_type=None),  # None
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if freeze_bn: self.freeze_bn()

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

        # 计算内部点与整个object的相似度
        feats = self.asp_ocr_head(feats, out_aux)

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

    def _load_pretrained_model(self, _pretrained_weights):
        checkpoint = torch.load(_pretrained_weights, map_location=torch.device('cpu'))
        state_dict = self.state_dict()
        model_dict = {}

        for k, v in checkpoint['state_dict'].items():
            model_dict[k] = v

        state_dict.update(model_dict)
        print('sucess')
        self.backbone.load_state_dict(state_dict, strict=False)

    def _load_checkpoint(self, checkpoint_path):
        print('start loading model: {}'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
        # checkpoint = checkpoint['state_dict']
        # from collections import OrderedDict
        # del_list_1 = ['cls_head.weight', 'aux_head.2.weight']
        del_list_2 = ["module.encoder_q.fc1.0.weight",
                      "module.encoder_q.fc1.0.bias",
                      "module.encoder_q.fc1.2.weight",
                      "module.encoder_q.fc1.2.bias"]

        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('module.encoder_q.'):
                # name = k[7:]  # 去掉 `module.`
                name = k.replace("module.encoder_q.backbone.", "", 1)
                if name in del_list_2:
                    continue
                new_state_dict[name] = v

        self.backbone.load_state_dict(new_state_dict, strict=True)

        print('Model Load Success!')