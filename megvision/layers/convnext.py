# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:convnext
    author: 12718
    time: 2022/1/12 14:54
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import megengine.module as M

class ConvNextBlock(M.Module):
    def __init__(self, in_ch):
        """
        Implementation of the block of ConvNext
        Parameters
        ----------
        in_ch (int): number of channels for input
        """
        super(ConvNextBlock, self).__init__()
        self.dw_conv = M.Conv2d(in_ch, in_ch, groups=in_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.norm = M.LayerNorm(in_ch)
        self.pw_conv1 = M.Conv2d(in_ch, in_ch*4, kernel_size=1, stride=1)
        self.activation = M.GELU()
        self.pw_conv2 = M.Conv2d(in_ch*4, in_ch, kernel_size=1, stride=1)

    def forward(self, x):
        identity = x
        net = self.dw_conv(x)
        net = self.norm(net)
        net = self.pw_conv1(net)
        net = self.activation(net)
        net = self.pw_conv2(net)
        net = net + identity
        return net