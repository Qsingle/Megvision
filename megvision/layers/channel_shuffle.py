# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:channel_shuffle
    author: 12718
    time: 2022/1/26 13:50
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import megengine as mge
import megengine.module as M
import megengine.functional as F

def channel_shuffle(x:mge.Tensor, groups:int):
    """
    Channel Shuffle introduced in ShuffleNet and ShuffleNetV2
    References:

    Parameters
    ----------
    x (Tensor): input tensor
    groups (int): number of groups

    Returns
    -------
        tensor after channel shuffle
    """
    bs, ch, h, w = x.shape
    assert ch % groups == 0, "The number of channel must be divided by groups"
    channel_per_groups = ch // groups
    #reshape
    x = x.reshape(bs, groups, channel_per_groups, h, w)
    #transpose
    x = F.transpose(x, (0, 2, 1, 3, 4))

    x = x.reshape(bs, -1, h, w)
    return x

class ChannelShuffle(M.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, groups=self.groups)