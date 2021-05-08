# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    @author: Zhongxi Qiu
    @create time: 2021/5/8 10:49
    @filename: encnet.py
    @software: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine.module import init

from ..classification.resnet import *

backbones = {
    "resnest50" : resnest50,
    "resnest101" : resnest101,
    "resnest14" : resnest14,
    "resnet50" : resnet50,
    "resnet101" : resnet101
}

class Encoding(M.Module):
    def __init__(self, K, D):
        super(Encoding, self).__init__()
        self.k = K
        self.d = D
        self.codes = mge.Parameter(mge.tensor(F.zeros((K, D))))
        self.scale = mge.Parameter(mge.tensor(F.zeros(K)))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / ((self.k * self.d) ** 0.5)
        init.uniform_(self.codes, -std, std)
        init.uniform_(self.scale, 0, 1)

    @staticmethod
    def scale_l2(x, c, s):
        s = s.reshape(1, 1, c.shape[0], 1)
        #expand
        x = F.expand_dims(x, axis=2)
        x = F.repeat(x, c.shape[0], axis=2)
        x = F.expand_dims(x, axis=3)
        x = F.repeat(x, c.shape[1], axis=3)
        c = F.expand_dims(F.expand_dims(c, axis=0), axis=1)
        out = (x - c) * s
        out = F.pow(out, out).sum(3)
        return out

    @staticmethod
    def aggregate(a, x, c):
        a = F.expand_dims(a, axis=3)
        # expand
        x = F.expand_dims(x, axis=2)
        x = F.repeat(x, c.shape[0], axis=2)
        x = F.expand_dims(x, axis=3)
        x = F.repeat(x, c.shape[1], axis=3)
        c = F.expand_dims(F.expand_dims(c, axis=0), axis=1)
        e = (x - c) * a
        e = e.sum(1)
        return e

    def forward(self, x):
        assert self.d == x.shape[1]
        bs, d = x.shape[:2]
        if x.ndim == 3:
            x = x.transpose(1, 2)
        elif x.ndim == 4:
            x = x.reshape(bs, d, -1).transpose(1, 2)
        else:
            raise ValueError("Unknown dim of input")
        a = self.scale_l2(x, self.codes, self.scale)
        e = self.aggregate(a, x, self.codes)
        return e

class EncModule(M.Module):
    def __init__(self, in_ch,n_classes, num_codes=32, se_loss=True, norm_layer=M.BatchNorm2d):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoder = M.Sequential(
            M.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, bias=False),
            norm_layer(in_ch),
            M.ReLU(),
            Encoding(num_codes,in_ch),
            M.BatchNorm1d(num_codes),
            M.ReLU()
        )
        self.fc = M.Sequential(
            M.Linear(in_ch, in_ch),
            M.Sigmoid()
        )
        if self.se_loss:
            self.se_layer = M.Linear(in_ch, n_classes)

    def forward(self, x):
        bs, c = x.size()[:2]
        en = self.encoder(x).mean(1)
        gamma = self.fc(en)
        y = gamma.reshape(bs, c, 1, 1)
        outputs = [F.relu(y*x + x)]
        if self.se_loss:
            outputs.append(self.se_layer(en))
        return outputs

class EncNet(M.Module):
    def __init__(self, in_ch, num_classes, num_codes=32, backbone="resnet50",
                 se_loss=True, norm_layer=M.BatchNorm2d, light_head=True,
                 laternel=True,
                 **kwargs):
        super(EncNet, self).__init__()
        dilations = [1, 1, 2, 4]
        strides = [1, 2, 1, 1]
        self.backbone = backbones[backbone](in_ch=in_ch, light_head=light_head,
                                            dilations=dilations, strides=strides,**kwargs)
        del self.backbone.fc
        del self.backbone.avg_pool
        self.conv5 = M.Sequential(
            M.Conv2d(2048, 512, kernel_size=1, stride=1),
            norm_layer(512),
            M.ReLU()
        )
        self.laternel = laternel
        if self.laternel:
            self.shortcut_c2 = M.Sequential(
                M.Conv2d(512, 512, kernel_size=1, stride=1),
                norm_layer(512),
                M.ReLU()
            )
            self.shortcut_c3 = M.Sequential(
                M.Conv2d(1024, 512, kernel_size=1, stride=1),
                norm_layer(512),
                M.ReLU()
            )
            self.fusion = M.Sequential(
                M.Conv2d(512*3, 512, kernel_size=1, stride=1),
                norm_layer(512),
                M.ReLU()
            )
        self.enc_module = EncModule(512, num_classes, num_codes=num_codes,
                                    se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = M.Sequential(
            M.Dropout(0.1),
            M.Conv2d(512, num_classes, kernel_size=1, stride=1)
        )


    def forward(self, x):
        net = self.backbone.conv1(x)
        net = self.backbone.max_pool(net)
        c1 = self.backbone.layer1(net)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)
        feat = self.conv5(c4)
        if self.laternel:
            c2 = self.shortcut_c2(c2)
            c3 = self.shortcut_c3(c3)
            feat = self.fusion(F.concat([feat, c2, c3], axis=1))

        outs = list(self.enc_module(feat))
        outs[0] = self.conv6(outs[0])
        outs[0] = F.vision.interpolate(outs[0], size=x.shape[2:], mode="bilinear", align_corners=True)
        return outs

if __name__ == "__main__":
    import numpy as np
    encoder = Encoding(32, 64)
    x = mge.tensor(np.random.normal(0, 1, (1, 64, 7, 7)))
    encoder(x)