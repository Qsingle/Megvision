# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:fim
    author: 12718
    time: 2022/9/2 10:48
    tool: PyCharm
"""
import megengine.module as M
import megengine.functional as F

from .channel_shuffle import ChannelShuffle

class SpatialModule(M.Module):
    def __init__(self, in_ch, out_ch):
        super(SpatialModule, self).__init__()
        hidden_state = out_ch * 3
        self.conv1 = M.Sequential(
            M.Conv2d(in_ch, hidden_state, 1, 1),
            M.ReLU()
        )
        self.branch1 = M.Conv2d(out_ch, out_ch, 3, 1, padding=1, dilation=1, groups=out_ch)
        self.branch2 = M.Conv2d(out_ch, out_ch, 3, 1, padding=2, dilation=2, groups=out_ch)
        self.branch3 = M.Conv2d(out_ch, out_ch, 3, 1, padding=4, dilation=4, groups=out_ch)
        self.shuffle = ChannelShuffle(3)
        self.fusion = M.Sequential(
            M.Conv2d(hidden_state, out_ch, 1, 1),
            M.ReLU()
        )

    def forward(self, x):
        net = self.conv1(x)
        splits = F.split(net, 3, axis=1)
        branch1 = self.branch1(splits[0])
        branch2 = self.branch2(splits[1])
        branch3 = self.branch3(splits[2])
        net = F.concat([branch1, branch2, branch3], axis=1)
        # net = F.relu(net)
        net = self.shuffle(net)
        net = self.fusion(net)
        return net

class FIM(M.Module):
    def __init__(self, in_ch1, in_ch2, hidden_state=16):
        super(FIM, self).__init__()
        self.conv1 = SpatialModule(in_ch1+in_ch2, hidden_state)
        self.conv2 = M.Sequential(
            M.Conv2d(hidden_state, 1, 1, 1),
            M.Sigmoid()
        )



    def forward(self, f1, f2):
        concat = F.concat([f1, f2], axis=1)
        net = self.conv1(concat)
        net = self.conv2(net)
        out_seg = net*f2 + f2
        return out_seg