# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:dac_rmp
    author: 12718
    time: 2021/12/8 16:47
    tool: PyCharm
"""
import megengine.module as M
import megengine.functional as F
import megengine.functional.vision as FV

__all__ = ["DAC", "RMP"]

class DAC(M.Module):
    def __init__(self, in_ch):
        """
            Implementation of Dense Atrous Convolution module
            References:
                "CE-Net: Context Encoder Network for 2D Medical Image Segmentation"
                <https://arxiv.org/pdf/1903.02740.pdf>
            Parameters
            ----------
            in_ch (int) number of input channels
        """
        super(DAC, self).__init__()
        self.dilate1 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dilate3 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilate5 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=5, dilation=5)
        self.conv1x1 = M.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.activation = M.ReLU()

    def forward(self, x):
        branch1 = self.activation(self.conv1x1(self.dilate5(self.dilate3(self.dilate1(x)))))
        branch2 = self.activation(self.conv1x1(self.dilate3(self.dilate1(x))))
        branch3 = self.activation(self.conv1x1(self.dilate3(x)))
        branch4 = self.activation(self.dilate1(x))
        out = x + branch1 + branch2 + branch3 + branch4
        return out

class RMP(M.Module):
    def __init__(self, in_ch):
        """
        Implementation of Residual Multi-kernel pooling
        References:
            "CE-Net: Context Encoder Network for 2D Medical Image Segmentation"
            <https://arxiv.org/pdf/1903.02740.pdf>
        Parameters
        ----------
        in_ch (int) number of input channels
        """
        super(RMP, self).__init__()
        self.conv1x1 = M.Conv2d(in_ch, 1, 1, 1, 0)

    def forward(self, x):
        size = x.shape[2:]
        branch1 = self.conv1x1(FV.interpolate(F.max_pool2d(x, 2, 2),
                                size=size, mode="bilinear", align_corners=True))
        branch2 = self.conv1x1(FV.interpolate(F.max_pool2d(x, 3, 3),
                                size=size, mode="bilinear", align_corners=True))
        branch3 = self.conv1x1(FV.interpolate(F.max_pool2d(x, 5, 6),
                             size=size, mode="bilinear", align_corners=True))
        branch4 = self.conv1x1(FV.interpolate(F.max_pool2d(x, 6, 6),
                                size=size, mode="bilinear", align_corners=True))
        out = F.concat([branch1, branch2, branch3, branch4, x], axis=1)
        return out