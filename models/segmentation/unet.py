# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/5/7 16:40
    @filename: unet.py
    @software: PyCharm
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine.jit import trace

from utils import *

class Downsample(M.Module):
    def __init__(self, in_ch, out_ch, convblock=DoubleConv,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU(), **kwargs):
        super(Downsample, self).__init__()
        self.conv = convblock(in_ch, out_ch, norm_layer=norm_layer, activation=activation, **kwargs)
        self.down = M.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feature = self.conv(x)
        down = self.down(feature)
        return feature, down

class Upsample(M.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, convblock=DoubleConv,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU(),
                 **kwargs):
        super(Upsample, self).__init__()
        self.conv = convblock(in_ch1 + out_ch, out_ch, norm_layer=norm_layer,
                              activation=activation, **kwargs)
        self.up_conv = M.Conv2d(in_ch2, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        up = F.vision.interpolate(x2, scale_factor=2.0, mode="bilinear")
        up = self.up_conv(up)
        cat = F.concat([x1, up], axis=1)
        out = self.conv(cat)
        return out


class Unet(M.Module):
    def __init__(self, in_ch, num_classes=3, convblock=DoubleConv,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU(),
                 layer_attention=False,**kwargs):
        super(Unet, self).__init__()
        self.down1 = Downsample(in_ch, 64, convblock=convblock, norm_layer=norm_layer,
                                activation=activation,**kwargs)
        self.down2 = Downsample(64, 128, convblock=convblock, norm_layer=norm_layer,
                                activation=activation,**kwargs)
        self.down3 = Downsample(128, 256, convblock=convblock, norm_layer=norm_layer,
                                activation=activation, **kwargs)
        self.down4 = Downsample(256, 512, convblock=convblock, **kwargs)
        self.down5 = convblock(512, 1024, norm_layer=norm_layer, activation=activation,**kwargs)
        self.up6 = Upsample(512, 1024, 512, convblock, norm_layer=norm_layer,
                            activation=activation,**kwargs)
        self.up7 = Upsample(256, 512, 256, convblock, norm_layer=norm_layer,
                            activation=activation, **kwargs)
        self.up8 = Upsample(128, 256, 128, convblock, norm_layer=norm_layer,
                            activation=activation, **kwargs)
        self.up9 = Upsample(64, 128, 64, convblock, norm_layer=norm_layer,
                            activation=activation,**kwargs)
        self.out_conv = M.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.layer_attention = layer_attention

    def forward(self, x):
        down1_f, down1 = self.down1(x)
        down2_f, down2 = self.down2(down1)
        down3_f, down3 = self.down3(down2)
        down4_f, down4 = self.down4(down3)
        down5 = self.down5(down4)
        up6 = self.up6(down4_f, down5)
        up7 = self.up7(down3_f, up6)
        up8 = self.up8(down2_f, up7)
        up9 = self.up9(down1_f, up8)
        if self.layer_attention:
            up9 = F.softmax(down1_f, axis=1)*up9 + up9
        out = self.out_conv(up9)
        return out

from megengine.jit import sublinear_memory_config as sublinear
config = sublinear.SublinearMemoryConfig()
@trace(sublinear_memory_config=config)
def test(model, x):
    out = model(x)
    print(out.shape)

if __name__ == "__main__":
    import numpy as np
    import time
    x = mge.tensor(np.random.normal(0, 1, (1, 3, 256, 256)))
    model = Unet(in_ch=3, convblock=R2Block, num_current=3)
    test(model, x)