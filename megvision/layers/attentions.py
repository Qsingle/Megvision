# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:attentions
    author: 12718
    time: 2021/12/2 15:51
    tool: PyCharm
"""

import megengine as mge
import megengine.module as M
import megengine.functional as F

__all__ = ["ChannelAttentionModule", "SpatialAttentionModule"]

class ChannelAttentionModule(M.Module):
    def __init__(self):
        """
            Implementation of Channel Attention Blocks introduced in DANet
            References:
                "Dual Attention Network for Scene Segmentation"
                <https://arxiv.org/pdf/1809.02983.pdf>
        """
        super(ChannelAttentionModule, self).__init__()
        self.gamma = F.zeros(1)

    def forward(self, x):
        b,c, h, w = x.shape
        proj_value = x.reshape(b, c, -1)
        proj_key = x.reshape(b, c, -1)
        prj_query = x.reshape(b, c, -1)
        key_query = F.matmul(prj_query, proj_key.transpose(0, 2, 1))
        key_query = F.broadcast_to(F.max(key_query, axis=-1, keepdims=True), key_query.shape) - key_query
        attention = F.softmax(key_query, axis=-1)
        value_attention = F.matmul(attention, proj_value)
        value_attention = value_attention.reshape(b, c, h, w)
        out = self.gamma*value_attention + x
        return out

class SpatialAttentionModule(M.Module):
    def __init__(self, in_ch):
        """
            Implementation of Spatial Attention Blocks introduced in DANet
            References:
                "Dual Attention Network for Scene Segmentation"
                <https://arxiv.org/pdf/1809.02983.pdf>
        """
        super(SpatialAttentionModule, self).__init__()
        self.key_proj = M.Conv2d(in_ch, in_ch // 8, 1, 1)
        self.query_proj = M.Conv2d(in_ch, in_ch // 8, 1, 1)
        self.value_proj = M.Conv2d(in_ch, in_ch, 1, 1)

        self.gamma = F.zeros(1)

    def forward(self, x):
        """

        Parameters
        ----------
        x (Tensor): the input feature map

        Returns
        -------
            attention map
        """
        bs, c, h, w = x.shape
        key = self.key_proj(x).reshape(bs, -1, h*w)
        query = self.query_proj(x).reshape(bs, -1, h*w)
        value = self.value_proj(x).reshape(bs, c, -1)
        query = query.transpose(0, 2, 1)
        energy = F.matmul(query, key)
        attention = F.softmax(energy)
        out = F.matmul(value, attention.transpose(0, 2, 1)).reshape(bs, c, h, w)
        out = out * self.gamma + x
        return out