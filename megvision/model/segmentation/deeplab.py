# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  deeplab.py
@Time    :  2021/8/26 16:27
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

import megengine as mge
import megengine.module as M
import megengine.functional as F

from megvision.model.classification.resnet import *
from megvision.model.classification import get_model
from megvision.comm.activation import Tanh
from megvision.layers import Conv2d
from megvision.layers import PixelShuffle
from megvision.layers import RCAB
from megvision.layers import SEB
from megvision.layers.sa import RSAB
from megvision.layers import FSL

from .build_model import SEGMENTATION_REGISTER


__all__ = ["DeeplabV3", "DeeplabV3Plus"]

class ImagePool(M.Module):
    def __init__(self, in_ch, out_ch):
        """
        Image pool introduced in DeeplabV3
        References:

        Args:
            in_ch (int): number of channels for input
            out_ch (int):  number of channels for output
        """
        super(ImagePool, self).__init__()
        self.pool = M.AdaptiveAvgPool2d((1, 1))
        self.conv = M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        net = self.pool(x)
        net = self.conv(net)
        return net

class ASPP(M.Module):
    def __init__(self, in_ch, rates):
        """
        Atrous Spatial Pyramid Pooling in Deeplab
        References:
            "SEMANTIC IMAGE SEGMENTATION WITH DEEP CONVOLUTIONAL NETS AND FULLY CONNECTED CRFS"<https://arxiv.org/pdf/1412.7062v3.pdf>
            "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs"<https://arxiv.org/abs/1606.00915>
            "Rethinking Atrous Convolution for Semantic Image Segmentation"<https://arxiv.org/abs/1706.05587>
            "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"<https://arxiv.org/abs/1802.02611>
        Args:
            in_ch (int): number of channels for input
            rates(List[int]): list of the astrous/dilation rate for each branchs
        """
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

        x5 = F.vision.interpolate(x5, size=x4.shape[2:], align_corners=True)
        _x = F.concat([x1, x2, x3, x4, x5], axis=1)
        net = self.concat_conv(_x)
        return net

@SEGMENTATION_REGISTER.register()
class DeeplabV3(M.Module):
    def __init__(self, in_ch=3, num_classes=3, backbone="resnet50", output_stride=16, **kwargs):
        """

        Args:
            in_ch:
            num_classes:
            backbone:
            output_stride:
            **kwargs:
        """
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
        not_imp = ["resnest269","resnet18","resnet34"]
        if backbone in not_imp:
            raise NotImplementedError("Not implement deeplabv3plus for {} as backbone".format(backbone))
        multi_grids = [1, 2, 4]
        self.backbone = get_model(backbone)(in_ch=in_ch, strides=strides, dilations=dilations,
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
        net = F.vision.interpolate(net, size=(h, w), align_corners=True)
        net = self.out_conv(net)
        return net

class Decoder(M.Module):
    def __init__(self, out_ch, low_ch, rates:list, rcab=False, seb=False, sa=False, layer_attention=False):
        super(Decoder, self).__init__()

        self.low_conv = M.Conv2d(low_ch, 48, kernel_size=1, stride=1)
        self.decoder_conv1 = M.Sequential(
            M.Conv2d(304, 256, kernel_size=3, stride=1, padding=1),
            M.Conv2d(256, out_ch, kernel_size=3, stride=1, padding=1)
        )
        self.layer_attention = layer_attention
        if sa:
            self.decoder_conv1 = RSAB(304, out_ch)
        if rcab:
            self.decoder_conv1 = RCAB(304, out_ch, stride=1)
        if seb:
            self.decoder_conv1 = SEB(304, out_ch)

    def forward(self, x, low_features):
        h = self.low_conv(low_features)
        net = F.vision.interpolate(x, size=h.shape[2:], mode="bilinear", align_corners=True)
        net = F.concat([net, h], axis=1)
        net = self.decoder_conv1(net)
        if self.layer_attention:
            net = F.softmax(low_features, axis=1) * net + net
        return net


@SEGMENTATION_REGISTER.register()
class DeeplabV3Plus(M.Module):
    def __init__(self, in_ch=3, num_classes=3, backbone="resnet50", output_stride=16,
                 layer_attention=False, super_reso=False, upscale_rate=4, middle=False,
                 rcab=False, seb=False, sa=False, **kwargs):
        """

        Args:
            in_ch:
            num_classes:
            backbone:
            output_stride:
            layer_attention:
            super_reso:
            upscale_rate:
            middle:
            rcab:
            **kwargs:
        """
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
        not_imp = ["resnest269","resnet18","resnet34"]
        if backbone in not_imp:
            raise NotImplementedError("Not implement deeplabv3plus for {} as backbone".format(backbone))
        self.backbone = get_model(backbone)(in_ch=in_ch, strides=strides, dilations=dilations,
                                                multi_grids=multi_grids, **kwargs)
        del self.backbone.fc
        del self.backbone.avgpool
        low_ch = 256
        if middle:
            low_ch = 512
        self.middle = middle
        self.aspp = ASPP(2048, rates=rates)
        self.seg_decoder = Decoder(256, low_ch, rates=rates,rcab=rcab, seb=seb,
                                   sa=sa, layer_attention=layer_attention)

        self.out_conv = M.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.super_reso = super_reso
        self.upscale_factor = upscale_rate
        if super_reso:
            self.decoder_sr = Decoder(256, low_ch=low_ch, rates=rates)
            # self.fa = M.Conv2d(num_classes, in_ch, 1, 1)
            self.sr = M.Sequential(
                M.Conv2d(256, 64, kernel_size=1, stride=1),
                M.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
                Tanh(),
                M.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                Tanh(),
                M.Conv2d(32, in_ch*(upscale_rate**2), kernel_size=3, stride=1, padding=1),
                PixelShuffle(upscale_rate)
            )
            # self.query = Query()
            self.query = FSL(sr_ch=in_ch, seg_ch=num_classes)

    def forward(self, x):
        net = self.backbone.conv1(x)
        net = self.backbone.bn1(net)
        net = self.backbone.relu(net)
        net = self.backbone.maxpool(net)
        net = self.backbone.layer1(net)
        hidden = net
        net = self.backbone.layer2(net)
        if self.middle:
            hidden = net
        net = self.backbone.layer3(net)
        net = self.backbone.layer4(net)
        net = self.aspp(net)
        seg_de = self.seg_decoder(net, hidden)
        if self.super_reso and self.training:
            decoder_sr = self.decoder_sr(net, hidden)
            decoder_sr = F.vision.interpolate(decoder_sr, size=x.shape[2:], mode="bilinear", align_corners=True)
            hr = self.sr(decoder_sr)
        net = F.vision.interpolate(seg_de, size=x.shape[2:], align_corners=True)
        net = self.out_conv(net)
        if self.super_reso:
            net = F.vision.interpolate(net, size=(x.shape[2]*self.upscale_factor, x.shape[3]*self.upscale_factor), align_corners=True)
        if self.super_reso and self.training:
            seg_weight = self.query(hr, net)
            qr_out = seg_weight*net + net
            return net, hr, qr_out
        return net

if __name__ == "__main__":
    import numpy as np
    x = mge.tensor(np.random.normal(0, 1, (1, 3, 256, 256)))
    model = DeeplabV3Plus(3, layer_attention=False, super_reso=True, upscale_rate=4)
    out = model(x)
    print(out[0].shape)