# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:sfa_aff
    author: 12718
    time: 2021/12/9 10:24
    tool: PyCharm
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import megengine.module as M
import megengine.functional as F

class SFA(M.Module):
    def __init__(self, in_ch):
        """
            Implementation of Scale-aware feature aggregation module
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
                <https://www.sciencedirect.com/science/article/pii/S1361841521000712>

            Parameters:
                in_ch (int): number of input channels
        """
        super(SFA, self).__init__()
        self.conv3x3_1 = M.Conv2d(in_ch, in_ch, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv3x3_2 = M.Conv2d(in_ch, in_ch, 3, stride=1, padding=3, dilation=3, bias=False)
        self.conv3x3_3 = M.Conv2d(in_ch, in_ch, 3, stride=1, padding=5, dilation=5, bias=False)

        self.conv3x3_12 = M.Sequential(
            M.Conv2d(in_ch*2, in_ch, 3, stride=1, padding=1, bias=False),
            M.ReLU()
        )

        self.conv3x3_23 = M.Sequential(
            M.Conv2d(in_ch*2, in_ch, 3, stride=1, padding=1, bias=False),
            M.ReLU()
        )

        self.conv1x1_12 = M.Conv2d(in_ch, 2, 1, bias=False)
        self.conv1x1_23 = M.Conv2d(in_ch, 2, 1, bias=False)

        self.out_conv = M.Sequential(
            M.Conv2d(in_ch, in_ch, 1, bias=False),
            M.ReLU()
        )

    def forward(self, x):
        d1 = self.conv3x3_1(x)
        d2 = self.conv3x3_2(x)
        d3 = self.conv3x3_3(x)

        f12 = F.concat([d1, d2], axis=1)
        f12 = self.conv3x3_12(f12)
        f12 = self.conv1x1_12(f12)

        f12 = F.softmax(f12, axis=1)
        w_as = F.split(f12, 2,  axis=1)

        w_f1 = w_as[0] * d1
        w_f2 = w_as[1] * d2

        w_f12 = w_f1 + w_f2

        f23 = F.concat([d2, d3], axis=1)
        f23 = self.conv3x3_23(f23)
        f23 = self.conv1x1_23(f23)
        f23 = F.softmax(f23, axis=1)
        w_bs = F.split(f23, 2, axis=1)

        w_f2_1 = w_bs[0] * d2
        w_f3 = w_bs[1] * d3

        w_f23 = w_f2_1 + w_f3
        out = w_f12 + w_f23 + x
        out = self.out_conv(out)
        return out


class AFF(M.Module):
    def __init__(self, in_ch, reduction=16):
        """
            Implementation of Adaptive feature fusion module
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
                <https://www.sciencedirect.com/science/article/pii/S1361841521000712>
            Parameters
            ----------
            in_ch (int): number of channels of input
            reduction (int): reduction rate for squeeze
        """
        super(AFF, self).__init__()
        in_ch1 = in_ch*2
        hidden_ch = (in_ch*2) // reduction
        self.se = M.Sequential(
            M.Conv2d(in_ch1, hidden_ch, 1),
            M.ReLU(),
            M.Conv2d(hidden_ch, in_ch1, 1),
            M.Sigmoid()
        )
        self.conv1x1 = M.Conv2d(in_ch1, in_ch, 1)

    def forward(self, x1, x2):
        """

        Parameters
        ----------
        x1 (Tensor): low level feature, (n,c,h,w)
        x2 (Tensor): high level feature, (n,c,h,w)

        Returns
        -------
            Tensor, fused feature
        """
        x12 = F.concat([x1, x2], axis=1)
        se = self.se(x12)
        se = self.conv1x1(se)
        se = F.adaptive_avg_pool2d(se, 1)
        se = F.sigmoid(se)
        w1 = se * x1
        out = w1 + x2
        return out

