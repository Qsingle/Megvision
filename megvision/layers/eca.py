# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  eca.py
@Time    :  2021/9/5 16:27
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

import megengine.module as M
import megengine.functional as F
import math

class ECAModule(M.Module):
    def __init__(self, channels, gamma=2, b=1,
                 activation=M.Sigmoid()):

        super(ECAModule, self).__init__()
        t = int(abs(math.log2(channels)/gamma + b / gamma))
        k = t if t % 2 else t+1
        self.avgpool = M.AdaptiveAvgPool2d(1)
        self.conv = M.Conv1d(1, 1, kernel_size=k,
                             padding=int(k/2), bias=False)
        self.activation = activation

    def forward(self, x):
        net = self.avgpool(x)
        net = F.squeeze(net, axis=-1)
        net = F.transpose(net, [0, 2, 1])
        net = self.conv(net)
        net = F.transpose(net, [0, 2, 1])
        net = F.expand_dims(net, axis=-1)
        net = self.activation(net)
        net = F.broadcast_to(net, x.shape)
        return net * x

if __name__ == "__main__":
    import megengine as mge
    x = mge.random.normal(size=(1, 64, 7, 7))
    model = ECAModule(64)
    model(x)