# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  rcab.py
@Time    :  2021/9/1 15:23
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

import megengine.module as M

from .layers import Conv2d, SEModule

class RCAB(M.Module):
    def __init__(self, in_ch, out_ch, stride=1,
                 norm_layer=M.BatchNorm2d,
                 activation=M.ReLU(),
                 attention_act=M.Sigmoid(),
                 reduction=16
                 ):
        """

        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            norm_layer (M.Module): normalization module
            activation (M.Module): activation function module
            attention_act (M.Module): attention activation function
            reduction (int): reduction rate
        """
        super(RCAB, self).__init__()
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

        self.se = SEModule(out_ch, reduction=reduction, norm_layer=norm_layer,
                           activation=activation, attention_act=attention_act)
        self.act = activation

    def forward(self, x):
        identify = x
        if self.downsample is not None:
            identify = self.downsample(x)
        net = self.conv(x)
        net = self.se(net) * net + identify
        return net