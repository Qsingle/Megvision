# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  seb.py
@Time    :  2021/9/2 16:30
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

import megengine.module as M

from .layers import Conv2d, SEModule


class SEB(M.Module):
    def __init__(self, in_ch, out_ch,
                 norm_layer=M.BatchNorm2d,
                 activation=M.ReLU(),
                 atten_act=M.Sigmoid(),
                 reduction=16
                 ):
        super(SEB, self).__init__()
        self.conv2 = M.Sequential(
            Conv2d(in_ch, out_ch, 3, stride=1, padding=1,
                   norm_layer=norm_layer, activation=activation),
            Conv2d(out_ch, out_ch, 3, stride=1, padding=1,
                   norm_layer=norm_layer, activation=None)
        )

        self.se = SEModule(out_ch, reduction=reduction, norm_layer=norm_layer,
                           activation=activation,
                           attention_act=atten_act)

        self.act = activation

    def forward(self, x):
        net = self.conv2(x)
        net = self.se(net) * net + net
        # net = self.act(net)
        return net