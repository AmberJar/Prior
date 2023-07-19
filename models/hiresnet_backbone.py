import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hiresnet_config import MODEL_CONFIGS
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from blocks import CA

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1, drop_path=0.,dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=planes, momentum=bn_momentum)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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


# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1, drop_path=0.,
#                  layer_scale_init_value=1e-6):
#         super().__init__()
#         # self.dwconv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, groups=inplanes)  # depthwise conv
#         self.dwconv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
#         self.norm = nn.BatchNorm2d(num_features=planes, momentum=0.1)
#         # self.ca = CA(planes)
#         # self.pwconv1 = nn.Linear(planes, 4 * planes)  # pointwise/1x1 convs, implemented with linear layers
#         self.pwconv1 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.act = nn.GELU()
#         # self.pwconv2 = nn.Linear(4 * planes, planes)
#         self.pwconv2 = nn.Conv2d(4 * planes, planes, kernel_size=1, bias=False)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(planes),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.downsample = downsample
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         input = x
#
#         x = self.dwconv(x)
#         x = self.norm(x)
#         # x = x + self.ca(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#
#         x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
#
#         if self.downsample is not None:
#             input = self.downsample(input)
#
#         x = input + self.drop_path(x)
#         return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1, drop_path=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        # self.dwconv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, groups=inplanes)  # depthwise conv
        self.dwconv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(num_features=planes, momentum=0.1)
        # self.pwconv1 = nn.Linear(planes, 4 * planes)  # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(4 * planes, planes)
        self.pwconv2 = nn.Conv2d(4 * planes, planes, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(planes),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.downsample = downsample
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.downsample is not None:
            input = self.downsample(input)

        x = input + self.drop_path(x)
        return x


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, bn_type=None, bn_momentum=0.1, drop_path=0.):
        super(HighResolutionModule, self).__init__()
        self.drop_path = drop_path
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        # self.ca_list = self._make_ca(self.num_branches, self.num_inchannels)
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels, bn_type=bn_type, bn_momentum=bn_momentum)
        self.fuse_layers = self._make_fuse_layers(bn_type=bn_type, bn_momentum=bn_momentum)
        self.relu = nn.GELU()

    def _make_ca(self, num_branches, num_inchannels):
        ca_list = []

        for i in range(num_branches):
            ca_list.append(CA(num_inchannels[i]))

        return nn.ModuleList(ca_list)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))

            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))

            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))

            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1, bn_type=None, bn_momentum=0.1):

        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.BatchNorm2d(num_features=self.num_inchannels[branch_index],
                               momentum=bn_momentum
                               ),
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
                bn_type=bn_type,
                bn_momentum=bn_momentum,
                drop_path=self.drop_path
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    bn_type=bn_type,
                    bn_momentum=bn_momentum,
                    drop_path=self.drop_path
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, bn_type, bn_momentum=0.1):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels, bn_type=bn_type, bn_momentum=bn_momentum)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, bn_type, bn_momentum=0.1):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.BatchNorm2d(num_features=num_inchannels[j], momentum=bn_momentum),
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.BatchNorm2d(num_features=num_inchannels[j], momentum=bn_momentum),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1, bias=False
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.BatchNorm2d(num_features=num_inchannels[j], momentum=bn_momentum),
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.GELU(),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])

            # y = y + self.ca_list[i](y)
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, bn_type, bn_momentum, drop_path_rate=0.5, **kwargs):
        self.inplanes = 64
        super(HighResolutionNet, self).__init__()

        self.depths = cfg['BLOCKS']
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur_dp_seq = 0

        if os.environ.get('full_res_stem'):
            print("using full-resolution stem with stride=1")
            stem_stride = 1
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stem_stride, padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=64, momentum=bn_momentum)
            self.relu = nn.GELU()
            self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, drop_path=dp_rates, cur_dp_seq=cur_dp_seq,
                                           bn_type=bn_type, bn_momentum=bn_momentum)
        else:
            stem_stride = 2
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=stem_stride, padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(num_features=64, momentum=bn_momentum)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=stem_stride, padding=1,
                                   bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=64, momentum=bn_momentum)
            self.relu = nn.GELU()
            self.layer1 = self._make_layer(Bottleneck, 64, 64, 4, drop_path=dp_rates, cur_dp_seq=cur_dp_seq,
                                           bn_type=bn_type, bn_momentum=bn_momentum)

        cur_dp_seq += self.depths[0]

        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]

        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels, bn_type=bn_type, bn_momentum=bn_momentum)

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, drop_path=dp_rates, cur_dp_seq=cur_dp_seq, bn_type=bn_type,
            bn_momentum=bn_momentum)

        cur_dp_seq += self.depths[1]

        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]

        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=bn_type, bn_momentum=bn_momentum)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, drop_path=dp_rates, cur_dp_seq=cur_dp_seq, bn_type=bn_type,
            bn_momentum=bn_momentum)

        cur_dp_seq += self.depths[2]

        # self.stage4_cfg = cfg['STAGE4']
        # num_channels = self.stage4_cfg['NUM_CHANNELS']
        # block = blocks_dict[self.stage4_cfg['BLOCK']]
        # num_channels = [
        #     num_channels[i] * block.expansion for i in range(len(num_channels))
        # ]
        # self.transition3 = self._make_transition_layer(
        #     pre_stage_channels, num_channels, bn_type=bn_type, bn_momentum=bn_momentum)
        #
        # self.stage4, pre_stage_channels = self._make_stage(
        #     self.stage4_cfg, num_channels, drop_path=dp_rates, cur_dp_seq=cur_dp_seq, multi_scale_output=True, bn_type=bn_type, bn_momentum=bn_momentum)
        #
        # cur_dp_seq += self.depths[3]

        if os.environ.get('keep_imagenet_head'):
            print('keep_imagenet_head, yes')
            self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels, drop_path=dp_rates, cur_dp_seq=cur_dp_seq,
                                               bn_type=bn_type, bn_momentum=bn_momentum)

    def _make_head(self, pre_stage_channels, drop_path, cur_dp_seq, bn_type, bn_momentum):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # print("pre_stage_channels: {}".format(pre_stage_channels))
        # print("head_channels: {}".format(head_channels))

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            drop_path=drop_path,
                                            cur_dp_seq=cur_dp_seq,
                                            bn_type=bn_type,
                                            bn_momentum=bn_momentum
                                            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels, momentum=bn_momentum),
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.GELU()
            )
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.BatchNorm2d(num_features=head_channels[3] * head_block.expansion, momentum=bn_momentum),
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.GELU()
        )
        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer, bn_type, bn_momentum):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.BatchNorm2d(num_features=num_channels_pre_layer[i], momentum=bn_momentum),
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False
                            ),
                            nn.GELU()
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.BatchNorm2d(num_features=inchannels, momentum=bn_momentum),
                            nn.Conv2d(
                                inchannels,
                                outchannels,
                                3,
                                2,
                                1,
                                bias=False
                            ),
                            nn.GELU()
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, drop_path=None, cur_dp_seq=None, bn_type=None,
                    bn_momentum=0.1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.BatchNorm2d(num_features=inplanes, momentum=bn_momentum),
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
            )
        cur_drop_path = drop_path[cur_dp_seq]
        cur_dp_seq += 1
        layers = []
        layers.append(
            block(inplanes, planes * block.expansion, stride, downsample, bn_type=bn_type, drop_path=cur_drop_path,
                  bn_momentum=bn_momentum))

        inplanes = planes * block.expansion
        planes = planes * block.expansion

        for i in range(1, blocks):
            cur_drop_path = drop_path[cur_dp_seq]
            layers.append(block(inplanes, planes, drop_path=cur_drop_path, bn_type=bn_type, bn_momentum=bn_momentum))
            cur_dp_seq += 1

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True, drop_path=None, cur_dp_seq=None, bn_type=None, bn_momentum=0.1):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            cur_drop_path = drop_path[cur_dp_seq]
            cur_dp_seq += 1

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    bn_type,
                    bn_momentum,
                    cur_drop_path
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        if os.environ.get('full_res_stem'):
            print('full_res_stem, yes')
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)

        return y_list

        # if os.environ.get('drop_stage4'):
        #     return y_list
        #
        # x_list = []
        # for i in range(self.stage4_cfg['NUM_BRANCHES']):
        #     if self.transition3[i] is not None:
        #         x_list.append(self.transition3[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # y_list = self.stage4(x_list)
        #
        #
        # if os.environ.get('keep_imagenet_head'):
        #     # Classification Head
        #     x_list = []
        #     y = self.incre_modules[0](y_list[0])
        #     x_list.append(y)
        #     for i in range(len(self.downsamp_modules)):
        #         y = self.incre_modules[i + 1](y_list[i + 1]) + \
        #             self.downsamp_modules[i](y)
        #         x_list.append(y)
        #
        #     y = self.final_layer(y)
        #     del x_list[-1]
        #     x_list.append(y)
        #
        #     return x_list
        # return y_list


def HRNetBackbone(arch):
    if arch == 'hrnet18':
        arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet18'],
                                     bn_type=None,
                                     bn_momentum=0.1)

    elif arch == 'hrnet32':
        arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet32'],
                                     bn_type=None,
                                     bn_momentum=0.1)

    elif arch == 'hrnet48':
        arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet48'],
                                     bn_type=None,
                                     bn_momentum=0.1)

    elif arch == 'hrnet64':
        arch_net = HighResolutionNet(MODEL_CONFIGS['hrnet64'],
                                     bn_type=None,
                                     bn_momentum=0.1)
    else:
        raise Exception('Architecture undefined!')

    return arch_net


if __name__ == '__main__':
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis
    from torchvision import models

    # model = HRNetBackbone(arch='hrnet48')
    model = models.vgg16().cuda()
    input_size = (1, 3, 1024, 1024)
    summary(model, input_size=input_size)

    x_in = torch.rand(size=input_size).cuda()

    f1 = FlopCountAnalysis(model, x_in)
    # print(f1.total())
    glops = f1.total() / 1E9 * 2
    print(glops)

    input = torch.tensor((3, 1024, 1024)).cuda()
    from torchstat import stat
    stat(model, input)
    # outs = model(x_in)
    #
    # for out in outs:
    #     print(out.shape)
