# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:sa
    author: 12718
    time: 2021/11/6 12:55
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import megengine as mge
import megengine.module as M
import megengine.functional as F

from layers import Conv2d


class SA(M.Module):
    def __init__(self):
        """
            Implementation of the Spatial Attention Used in SA-UNet.
            References:
                SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation
                <https://arxiv.org/ftp/arxiv/papers/2004/2004.03696.pdf>

        """
        super(SA, self).__init__()
        self.conv = M.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        mxp = F.max(x, axis=1, keepdims=True)
        avp = F.mean(x, axis=1, keepdims=True)
        cat = F.concat([mxp, avp], axis=1)
        atten = self.conv(cat)
        atten = F.sigmoid(atten)
        return x * atten


class RSAB(M.Module):
    def __init__(self, in_ch:int, out_ch:int, stride=1,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU()):
        """

        Parameters
        ----------
        in_ch
        out_ch
        stride
        norm_layer
        activation
        """
        super(RSAB, self).__init__()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            if stride != 1:
                self.downsample = M.Sequential(
                    M.AvgPool2d(kernel_size=3, stride=stride, padding=(3 // stride)),
                    Conv2d(in_ch, out_ch, ksize=1, stride=1, norm_layer=norm_layer,
                           activation=None)
                )
            else:
                self.downsample = Conv2d(in_ch, out_ch, ksize=1, stride=1, norm_layer=norm_layer,
                                         activation=None)
        self.conv = M.Sequential(
            Conv2d(in_ch, out_ch, ksize=3, stride=stride,
                   padding=1, norm_layer=norm_layer,
                   activation=activation),
            Conv2d(out_ch, out_ch, ksize=3, stride=1, padding=1,
                   norm_layer=norm_layer, activation=None)
        )

        self.sa = SA()
        self.act = activation

    def forward(self, x):
        identify = x if self.downsample is None else self.downsample(x)
        net = self.conv(x)
        net = self.sa(net) * net
        net = net + identify
        return net

if __name__ == "__main__":
    x = mge.random.normal(size=(1, 256, 128, 128))
    m = SA()
    m(x)

