# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/5/7 14:35
    @filename: utils.py.py
    @software: PyCharm
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import megengine as mge
import megengine.module as M
import megengine.functional as F

__all__ = ["Conv2d", "RConv", "R2Block", "DoubleConv", "SplAtConv2d", "SEModule"]

class Conv2d(M.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0, groups=1, dilation=1,
                 bias=False, norm_layer=None, activation=None, **kwargs):
        """
        Conv with normalization and activation
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels (or you can see it as how many of conv kernel)
            ksize (Union[int, tuple]): kernel size of conv, default is 1
            stride (Union[int, tuple]): stride of conv, default is 1
            padding (Union[int, tuple]): padding size, default is 0
            groups (int): number of groups for conv, default is 1,
            dilation (int): dilation rate of conv, default is 1
            bias (bool): whether use bias
            norm_layer (M.Module): normalization layer
            activation (M.Module): nonlinear activation function
        """
        super(Conv2d, self).__init__()
        self.conv = M.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, dilation=dilation,
                             padding=padding, groups=groups, bias=bias, **kwargs)
        self.norm_layer = None
        if norm_layer is not None:
            self.norm_layer = norm_layer(out_ch)
        self.activation = activation
        if isinstance(activation, M.PReLU):
            self.activation = activation(out_ch)

    def forward(self, inputs):
        net = self.conv(inputs)
        if self.norm_layer is not None:
            net = self.norm_layer(net)
        if self.activation is not None:
            net = self.activation(net)
        return net

class DoubleConv(M.Module):
    def __init__(self, in_ch, out_ch, norm_layer=M.BatchNorm2d,
                 activation=M.ReLU(), **kwargs):
        """
        Double convolutional layer in Unet.
        References:
            "U-Net: Convolutional Networks for Biomedical Image Segmentation"
            <https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28>
        Args:
            in_ch (int): number of input channels
            norm_layer (M.Module): normalization layer
            activation (M.Module): activation layer
        """
        super(DoubleConv, self).__init__()
        self.conv1 = Conv2d(in_ch, out_ch, ksize=3, stride=1, padding=1,
                            norm_layer=norm_layer, activation=activation, **kwargs)
        self.conv2 = Conv2d(out_ch, out_ch, ksize=3, stride=1, padding=1,
                            norm_layer=norm_layer, activation=activation, **kwargs)


    def forward(self, x):
        net = self.conv1(x)
        net = self.conv2(net)
        return net

class RConv(M.Module):
    def __init__(self, out_ch, num_recurrent=2, norm_layer=M.BatchNorm2d,
                 activation=M.ReLU(), **kwargs):
        """
        Recurrent conv.
        References:
            "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation"
            <https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf>
        Args:
            out_ch (int): number of output channels
            num_recurrent (int): times or recurrent
            norm_layer (nn.Module): Normalization module
            activation (nn.Module): Non-linear activation module
        Args:
            out_ch (int): number of output channels
            num_recurrent (int): times of recurrent
            norm_layer (M.Module): normalization module
            activation (M.Module): non-linear module
        """
        super(RConv, self).__init__()
        self.conv = Conv2d(out_ch, out_ch, ksize=3, stride=1, padding=1,
                           norm_layer=norm_layer, activation=activation, **kwargs)
        self.num_current = num_recurrent

    def forward(self, x):
        x1 = x
        for i in range(self.num_current):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x+x1)
        return x1

class R2Block(M.Module):
    def __init__(self, in_ch, out_ch, num_current=2,norm_layer=M.BatchNorm2d,
                 activation=M.ReLU(), **kwargs):
        """
        R2Block in R2Unet
        References:
            "Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation"
            <https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf>
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            num_recurrent (int): times of recurrent
            norm_layer (M.Module): normalization module
            activation (M.Module): non-linear module
        """
        super(R2Block, self).__init__()
        self.conv1x1 = M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv = M.Sequential(
            RConv(out_ch, num_recurrent=num_current, norm_layer=norm_layer,
                  activation=activation, **kwargs),
            RConv(out_ch, num_recurrent=num_current, norm_layer=norm_layer,
                  activation=activation, **kwargs)
        )

    def forward(self, x):
        identify = self.conv1x1(x)
        net = self.conv(identify)
        net = net + identify
        return net

