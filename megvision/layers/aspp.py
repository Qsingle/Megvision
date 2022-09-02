# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:aspp
    author: 12718
    time: 2021/11/27 12:50
    tool: PyCharm
"""
import megengine.module as M
import megengine.functional as F
from .layers import Conv2d

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