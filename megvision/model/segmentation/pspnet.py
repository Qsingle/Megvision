# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:pspnet
    author: 12718
    time: 2021/11/19 16:40
    tool: PyCharm
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import megengine.module as M
import megengine.functional as F

from megvision.model.classification.resnet import *
from megvision.model.classification.build import get_model
from megvision.layers import PSPModule

from .build_model import SEGMENTATION_REGISTER

__all__ = ["PSPNet"]

SUPPORTED_BACKBONES = ["resnet50", "resnest50", "resnet101",
                       "resnest101", "resnet152", "resnest200", "resnest269"]

SEGMENTATION_REGISTER.register()
class PSPNet(M.Module):
    def __init__(self, in_ch=3, num_classes=20, backbone="resnet50", aux=True, dropout=0.1,
                 bins=[1, 2, 3, 6],pretrained=True):
        """
        Implementation of PSPNet.

        References:
         "Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>

        Parameters
        ----------
        in_ch (int): number of channels for input
        num_classes (int): number of classes
        backbone (str): name of backbone
        aux (bool): whether use auxiliary loss
        dropout (float): dropout rate
        bins (list): sizes of the spatial in psp module
        pretrained (bool): whether use pretrained backbone
        """
        super(PSPNet, self).__init__()
        assert backbone in SUPPORTED_BACKBONES, "The name of backbone must in {}, but got {}".format(
            SUPPORTED_BACKBONES,
            backbone)
        self.backbone = get_model(backbone)(in_ch=in_ch, pretrained=pretrained,
                                            strides=[1, 2, 2, 1], dilations=[1, 1, 2, 4])
        del self.backbone.fc
        fea_dim = 2048
        self.psp = PSPModule(in_ch=fea_dim, out_ch=fea_dim // len(bins), sizes=bins)
        fea_dim = fea_dim * 2
        self.cls = M.Sequential(
            M.Conv2d(fea_dim, 512, kernel_size=3, stride=1, padding=1, bias=False),
            M.BatchNorm2d(512),
            M.ReLU(),
            M.Dropout(dropout),
            M.Conv2d(512, num_classes, 1, 1)
        )
        self.aux = aux
        if aux:
            self.aux_cls = M.Sequential(
                M.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1,bias=False),
                M.BatchNorm2d(256),
                M.ReLU(),
                M.Dropout(dropout),
                M.Conv2d(256, num_classes, 1, 1)
            )

    def forward(self, x):
        h, w = x.shape[2:]
        net = self.backbone.conv1(x)
        net = self.backbone.bn1(net)
        net = self.backbone.relu(net)
        net = self.backbone.maxpool(net)

        net = self.backbone.layer1(net)
        net = self.backbone.layer2(net)
        net = self.backbone.layer3(net)
        if self.aux and self.training:
            aux_out = self.aux_cls(net)
            aux_out = F.vision.interpolate(aux_out, size=(h,w), mode="bilinear", align_corners=True)
        net = self.backbone.layer4(net)
        net = self.psp(net)
        net = self.cls(net)
        net = F.vision.interpolate(net, size=(h, w), mode="bilinear", align_corners=True)
        if self.training and self.aux:
            return net, aux_out
        else:
            return net

if __name__ == "__main__":
    import megengine as mge
    model = PSPNet(backbone="resnest101", pretrained=False)
    out = model(mge.random.normal(size=(1, 3, 224, 224)))
    print(out[0].shape)