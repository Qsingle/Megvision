# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:pspmodule
    author: 12718
    time: 2021/11/19 14:35
    tool: PyCharm
"""


import megengine.module as M
import megengine.functional as F

from layers import Conv2d, AdaptiveAvgPool2d

class PSPModule(M.Module):
    def __init__(self, in_ch, out_ch, sizes=(1, 2, 3, 6)):
        """
        Implementation of Pyramid Spatial Pooling Module introduced in
        "Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>
        Parameters
        ----------
        in_ch (int): number of channels for output
        out_ch (int): number of channels for each branch's output
        sizes (list): size of outputs, default is (1, 2, 3, 6)
        """
        super(PSPModule, self).__init__()
        self.stages = [
            M.Sequential(
                AdaptiveAvgPool2d(size),
                Conv2d(in_ch, out_ch)
            ) for size in sizes
        ]

    def forward(self, x):
        h,w = x.shape[2:]
        outs = [x]
        for stage in self.stages:
            upsample = F.vision.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=True)
            outs.append(upsample)
        out = F.concat(outs, axis=1)
        return out

if __name__ == "__main__":
    import megengine as mge
    PSPModule(64, 32)(mge.random.normal(size=(1, 64, 64//4, 36)))