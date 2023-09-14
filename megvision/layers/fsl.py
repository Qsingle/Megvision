# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:fsl
    author: 12718
    time: 2022/3/2 15:41
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import megengine as mge
import megengine.module as M
import megengine.functional as F

class FSL(M.Module):
    def __init__(self, sr_ch, seg_ch, hidden_state=32):
        """
        Fine-grained semantic learning module
        Parameters
        ----------
        seg_ch (int): numcer of channels for segmentation features
        sr_ch (int): number of channels for super-resolution
        """
        super(FSL, self).__init__()
        self.conv1=M.Sequential(
            M.Conv2d(sr_ch+seg_ch, hidden_state, 1, 1),
            M.ReLU()
        )
        self.conv_sp1 = M.Conv2d(hidden_state, hidden_state,
                                 (7, 1), padding=(3, 0), bias=False)
        self.conv_sp2 = M.Conv2d(hidden_state, hidden_state,
                                 (1, 7), padding=(0, 3), bias=False)
        self.fusion = M.Sequential(
            M.ReLU(),
            M.Conv2d(hidden_state, seg_ch, 1, 1),
            M.Sigmoid()
        )

    def forward(self, sr_fe, seg_fe):
        concat = F.concat([sr_fe, seg_fe], axis=1)
        conv = self.conv1(concat)
        sp1 = self.conv_sp1(conv)
        sp2 = self.conv_sp2(conv)
        seg_fusion = self.fusion(sp1+sp2)
        return seg_fusion