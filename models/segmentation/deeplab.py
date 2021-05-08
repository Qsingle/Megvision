# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/5/7 17:32
    @filename: deeplab.py
    @software: PyCharm
"""
import megengine as mge
import megengine.module as M
import megengine.functional as F

from ..classification.resnet import *
from .utils import Conv2d

backbone_dict = {
    "resnet50" : resnet50,
    "resnet101" : resnet101,
    "resnest101" : resnest101,
    "resnest50" : resnest50
}

class ImagePool(M.Module):
    def __init__(self, in_ch, out_ch):
        super(ImagePool, self).__init__()
        self.pool = M.AdaptiveAvgPool2d((1, 1))
        self.conv = M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        net = self.pool(x)
        net = self.conv(net)
        return net

class ASPP(M.Module):
    def __init__(self, in_ch, rates):
        super(ASPP, self).__init__()
        self.branch1 = Conv2d(in_ch, 256, dilation=rates[0])
        self.branch2 = Conv2d(in_ch, 256, ksize=3, stride=1, padding=rates[1], dilation=rates[1])
        self.branch3 = Conv2d(in_ch, 256, ksize=3, stride=1, padding=rates[2], dilation=rates[2])
        self.branch4 = Conv2d(in_ch, 256, ksize=3, stride=1, padding=rates[3], dilation=rates[3])
        self.branch5 = ImagePool(in_ch, 256)

        self.concat_conv = Conv2d(256*5, 256)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x5 = self.branch5(x)
        x5 = F.vision.interpolate(x5, size=x4.shape[2:])
        _x = F.concat([x1, x2, x3, x4, x5], axis=1)
        net = self.concat_conv(_x)
        return net

class DeeplabV3(M.Module):
    def __init__(self, in_ch=3, num_classes=3, backbone="resnet50", output_stride=16, **kwargs):
        super(DeeplabV3, self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 6, 12, 18]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 12, 24, 36]
        else:
            raise ValueError("Unknown output stride, except 16 or 8 but got {}".format(output_stride))
        multi_grids = [1, 2, 4]
        self.backbone = backbone_dict[backbone](in_ch=in_ch, strides=strides, dilations=dilations,
                                                multi_grids=multi_grids, **kwargs)
        del self.backbone.fc
        del self.backbone.avgpool

        self.aspp = ASPP(2048, rates=rates)
        self.out_conv = M.Conv2d(256, num_classes, kernel_size=1, stride=1)


    def forward(self, x):
        net = self.backbone.conv1(x)
        net = self.backbone.bn1(net)
        net = self.backbone.relu(net)
        net = self.backbone.maxpool(net)
        net = self.backbone.layer1(net)
        net = self.backbone.layer2(net)
        net = self.backbone.layer3(net)
        net = self.backbone.layer4(net)
        h,w = x.shape[2:]
        net = self.aspp(net)
        net = F.vision.interpolate(net, size=(h, w))
        net = self.out_conv(net)
        return net

class DeeplabV3Plus(M.Module):
    def __init__(self, in_ch=3, num_classes=3, backbone="resnet50", output_stride=16,
                 layer_attention=False,**kwargs):
        super(DeeplabV3Plus, self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 6, 12, 18]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 1, 2]
            rates = [1, 12, 24, 36]
        else:
            raise ValueError("Unknown output stride, except 16 or 8 but got {}".format(output_stride))
        multi_grids = [1, 2, 4]
        self.backbone = backbone_dict[backbone](in_ch=in_ch, strides=strides, dilations=dilations,
                                                multi_grids=multi_grids, **kwargs)
        del self.backbone.fc
        del self.backbone.avgpool

        self.low_conv = M.Conv2d(256, 48, kernel_size=1, stride=1)
        self.aspp = ASPP(2048, rates=rates)
        self.decoder_conv1 = M.Sequential(
            M.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            M.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )
        self.out_conv = M.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.layer_attention = layer_attention


    def forward(self, x):
        net = self.backbone.conv1(x)
        net = self.backbone.bn1(net)
        net = self.backbone.relu(net)
        net = self.backbone.maxpool(net)
        net = self.backbone.layer1(net)
        hidden = net
        net = self.backbone.layer2(net)
        net = self.backbone.layer3(net)
        net = self.backbone.layer4(net)
        h,w = hidden.shape[2:]
        hidden_conv = self.low_conv(hidden)
        net = self.aspp(net)
        net = F.vision.interpolate(net, size=(h, w))
        net = F.concat([net, hidden_conv], axis=1)
        net = self.decoder_conv1(net)
        if self.layer_attention:
            net = F.softmax(hidden) * net + net
        net = F.vision.interpolate(net, size=x.shape[2:])
        net = self.out_conv(net)
        return net

if __name__ == "__main__":
    import numpy as np
    x = mge.tensor(np.random.normal(0, 1, (1, 3, 256, 256)))
    model = DeeplabV3Plus(3, layer_attention=True)
    out = model(x)
    print(out.shape)