# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:espnet
    author: 12718
    time: 2021/11/19 14:11
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import megengine as mge
import megengine.module as M
import megengine.functional as F

from megvision.model.classification.espnets import *
from megvision.model.classification.build import get_model
from megvision.layers import Conv2d, EESP
from megvision.comm.activation import Tanh
try:
    from megengine.module import PixelShuffle
except Exception as e:
    from megvision.layers import PixelShuffle
from megvision.layers import FIM

from .build_model import SEGMENTATION_REGISTER

__all__ = ["ESPNetV2_Seg"]

class PSP(M.Module):
    def __init__(self, in_ch, out_ch=1024, sizes=(1, 2, 4, 8)):
        """
        Implementation of the pspmodule in ESPNetV2 follows as
        <https://github.com/sacmehta/ESPNetv2/blob/b78e323039908f31347d8ca17f49d5502ef1a594/segmentation/cnn/cnn_utils.py>
        Parameters
        ----------
        in_ch (int): number of chanenls for input
        out_ch (int): number of channels for output
        sizes (list): kernel sizes for convs, but not used in this implementation.
        """
        super(PSP, self).__init__()
        self.stages = [M.Conv2d(in_ch, in_ch, 3, 1, groups=in_ch, padding=1, bias=False) for size in sizes]
        self.project = M.Sequential(
            Conv2d(in_ch*(len(sizes)+1), out_ch, 1, 1, activation=M.PReLU(out_ch))
        )

    def forward(self, feats):
        h, w = feats.shape[2:]
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding=1)
            upsampled = F.vision.interpolate(stage(feats), size=(h,w), mode="bilinear", align_corners=True)
            out.append(upsampled)
        out = self.project(F.concat(out, axis=1))
        return out

@SEGMENTATION_REGISTER.register()
class ESPNetV2_Seg(M.Module):
    def __init__(self, in_ch=3, num_classes=20, backbone="espnetv2_s_1_0", s=1, pretrained=True, super_reso=False, upscale_rate=2):
        """
        Implementation espnetv2 for segmentation
        References:
        official pytorch code: https://github.com/sacmehta/ESPNetv2
        "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network" <https://arxiv.org/pdf/1811.11431.pdf>
        "ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation" <https://arxiv.org/pdf/1803.06815.pdf>

        Parameters
        ----------
        in_ch (int): number of channels for input
        num_classes (int): number of classes
        backbone (str): name of the backbone, choices=["espnetv2_s_0_5", "espnetv2_s_1_0", "espnetv2_s_1_25",
                                        "espnetv2_s_1_5", "espnetv2_s_2_0"]
        s (float): scale rate for the espnet
        pretrained (bool): whether use pretrained backbone
        super_reso (bool): whether use super resolution
        upscale_rate (int): upscale rate for super resolution
        """
        super(ESPNetV2_Seg, self).__init__()
        supported_backbones = ["espnetv2_s_0_5", "espnetv2_s_1_0", "espnetv2_s_1_25",
                                        "espnetv2_s_1_5", "espnetv2_s_2_0"]
        assert backbone in supported_backbones, "Name of the backbone must in {}, but got {}".format(supported_backbones, backbone)
        self.backbone = get_model(backbone)(pretrained=pretrained, in_ch=in_ch, num_classes=1000)
        del self.backbone.classifier
        del self.backbone.level5
        del self.backbone.level5_0
        ch = self.backbone.level4[-1].module_act.num_parameters
        out_ch = self.backbone.level3[-1].module_act.num_parameters
        self.proj_l4_c = Conv2d(ch, out_ch, activation=M.PReLU(out_ch))

        if s <= 0.5:
            p = 0.1
        else:
            p = 0.2
        pspSize = 2*self.backbone.level3[-1].module_act.num_parameters
        self.pspMod = M.Sequential(
            EESP(pspSize, pspSize//2, stride=1, K=4, r_lim=7),
            PSP(pspSize // 2, pspSize//2)
        )

        self.project_l3 = M.Sequential(
            M.Dropout(p),
            M.Conv2d(pspSize//2, num_classes, 1, 1, bias=False)
        )
        self.act_l3 = M.Sequential(
            M.BatchNorm2d(num_classes),
            M.PReLU(num_classes)
        )
        ch = self.backbone.level2_0.activation.num_parameters + num_classes
        self.project_l2 = Conv2d(ch, num_classes, 1, 1, activation=M.PReLU(num_classes))
        ch = self.backbone.level1.activation.num_parameters + num_classes
        self.project_l1 = M.Sequential(
            M.Dropout(p),
            M.Conv2d(ch, num_classes, 1, 1, bias=False)
        )
        self.super_reso = super_reso
        if super_reso:
            self.sr_proj_l4_c = Conv2d(self.backbone.level4[-1].module_act.num_parameters, out_ch, activation=M.PReLU(out_ch))
            self.sr_pspMod = M.Sequential(
                EESP(pspSize, pspSize // 2, stride=1, K=4, r_lim=7),
                PSP(pspSize // 2, pspSize // 2)
            )
            self.sr_project_l3 = M.Sequential(
                M.Dropout(p),
                M.Conv2d(pspSize//2, in_ch, 1, 1, bias=False)
            )
            self.sr_act_l3 = M.Sequential(
                M.BatchNorm2d(in_ch),
                M.PReLU(in_ch)
            )
            ch = self.backbone.level2_0.activation.num_parameters + in_ch
            self.sr_project_l2 = Conv2d(ch, in_ch, 1, 1, activation=M.PReLU(in_ch))
            ch = self.backbone.level1.activation.num_parameters + in_ch
            self.sr_project_l1 = M.Sequential(
                M.Dropout(p),
                M.Conv2d(ch, in_ch, 1, 1, bias=False)
            )
            self.upscale_rate = upscale_rate
            self.super_conv = M.Sequential(
                M.Conv2d(in_ch, 64, kernel_size=5, stride=1, padding=2, bias=False),
                Tanh(),
                M.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                Tanh(),
                M.Conv2d(32, (upscale_rate ** 2) * in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                PixelShuffle(upscale_factor=upscale_rate)
            )
            self.interaction = FIM(in_ch, num_classes, hidden_state=16)

    def hierarchicalUpsample(self, x, upscale_range=3):
        for _ in range(upscale_range):
            x = F.vision.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        return x

    def forward(self, x):
        out_l1 = self.backbone.level1(x)
        if not self.backbone.reinf:
            del x
            x = None
        out_l2 = self.backbone.level2_0(out_l1, x)

        out_l3_0 = self.backbone.level3_0(out_l2, x)
        for i, layer in enumerate(self.backbone.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        outl4_0 = self.backbone.level4_0(out_l3, x)
        for i, layer in enumerate(self.backbone.level4):
            if i == 0:
                out_l4 = layer(outl4_0)
            else:
                out_l4 = layer(out_l4)
        out_l4_proj = self.proj_l4_c(out_l4)
        up_l4_to_l3 = F.vision.interpolate(out_l4_proj, size=out_l3.shape[2:], mode="bilinear", align_corners=True)
        merged_l3_upl4 = F.concat([out_l3, up_l4_to_l3], axis=1)
        merged_l3_upl4 = self.pspMod(merged_l3_upl4)
        proj_l3 = self.project_l3(merged_l3_upl4)
        act_l3 = self.act_l3(proj_l3)
        up_l3_to_l2 = F.vision.interpolate(act_l3, size=out_l2.shape[2:], mode="bilinear", align_corners=True)
        merged_l2 = F.concat([out_l2, up_l3_to_l2], axis=1)
        proj_l2 = self.project_l2(merged_l2)
        up_l2_to_l1 = F.vision.interpolate(proj_l2, size=out_l1.shape[2:], mode="bilinear", align_corners=True)
        merged_l1 = F.concat([out_l1, up_l2_to_l1], axis=1)
        out_proj_l1 = self.project_l1(merged_l1)
        out_proj_l1 = F.vision.interpolate(out_proj_l1, scale_factor=2, mode="bilinear", align_corners=True)
        if self.training:
            proj_l3 = self.hierarchicalUpsample(proj_l3)
        if self.super_reso:
            out_proj_l1 = F.vision.interpolate(out_proj_l1, scale_factor=self.upscale_rate, mode="bilinear",
                                               align_corners=True)
            if self.training:
                proj_l3 = F.vision.interpolate(proj_l3, scale_factor=self.upscale_rate, mode="bilinear", align_corners=True)
                sr_out_l4_proj = self.sr_proj_l4_c(out_l4)
                sr_up_l4_to_l3 = F.vision.interpolate(sr_out_l4_proj, size=out_l3.shape[2:], mode="bilinear", align_corners=True)
                sr_merged_l3_upl4 = F.concat([out_l3, sr_up_l4_to_l3], axis=1)
                sr_merged_l3_upl4 = self.sr_pspMod(sr_merged_l3_upl4)
                sr_proj_l3 = self.sr_project_l3(sr_merged_l3_upl4)
                sr_act_l3 = self.sr_act_l3(sr_proj_l3)
                sr_up_l3_to_l2 = F.vision.interpolate(sr_act_l3, size=out_l2.shape[2:], mode="bilinear", align_corners=True)
                sr_merged_l2 = F.concat([out_l2, sr_up_l3_to_l2], axis=1)
                sr_proj_l2 = self.sr_project_l2(sr_merged_l2)
                sr_up_l2_to_l1 = F.vision.interpolate(sr_proj_l2, size=out_l1.shape[2:], mode="bilinear", align_corners=True)
                sr_merged_l1 = F.concat([out_l1, sr_up_l2_to_l1], axis=1)
                sr_out_proj_l1 = self.sr_project_l1(sr_merged_l1)
                sr_out_proj_l1 = F.vision.interpolate(sr_out_proj_l1, scale_factor=2, mode="bilinear", align_corners=True)
                sr = self.super_conv(sr_out_proj_l1)
                qr_proj_l1 = self.interaction(sr, out_proj_l1) * out_proj_l1 + out_proj_l1

        if self.training:
            if self.super_reso:
                return out_proj_l1, proj_l3, sr, qr_proj_l1
            return out_proj_l1, proj_l3
        else:
            return out_proj_l1



if __name__ == "__main__":
    model = ESPNetV2_Seg(3, num_classes=20)
    out1, out2 = model(mge.random.normal(size=(1, 3, 224, 224)))
    print(out1.shape)
