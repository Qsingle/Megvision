# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  layers.py
@Time    :  2021/8/26 14:04
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import megengine as mge
import megengine.functional as F
import megengine.module as M
# from megengine.functional.nn import _pair

from comm.tuple_functools import _pair

__all__ = ["Conv2d", "SeparableConv2d", "Dropout2d", "SplAtConv2d", "SEModule", "AdaptiveAvgPool2d"]


class AdaptiveAvgPool2d(M.Module):
    def __init__(self, out_size):
        super(AdaptiveAvgPool2d, self).__init__()
        self.out_size = out_size

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, self.out_size)

class Conv2d(M.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0,
                 bias=False, dilation=1, groups=1, norm_layer=M.BatchNorm2d,
                 activation=M.ReLU(), gn_groups=32, dropout=0.0, drop_block_size=0, **kwargs):
        super(Conv2d, self).__init__()
        ksize = _pair(ksize)
        stride = _pair(stride)
        padding = _pair(padding)

        self.conv = M.Conv2d(in_ch, out_ch, ksize, stride=stride,
                             padding=padding, dilation=dilation,
                             groups=groups, bias=bias, **kwargs
                             )
        self.dropout = dropout
        if dropout > 0.0:
            if drop_block_size > 0:
                self.drop_block = Dropout2d(dropout, kernel_size=drop_block_size)
            else:
                self.drop_block = M.Dropout(dropout)
        self.norm_layer = None
        if norm_layer is not None:
            if isinstance(norm_layer, M.GroupNorm):
                self.norm_layer = norm_layer(gn_groups, out_ch)
            else:
                self.norm_layer = norm_layer(out_ch)

        self.activation = activation

    def forward(self, x):
        net = self.conv(x)
        if self.dropout > 0.0:
            net = self.drop_block(net)
        if self.norm_layer is not None:
            net = self.norm_layer(net)
        if self.activation is not None:
            net = self.activation(net)
        return net

class SEModule(M.Module):
    def __init__(self, channels, reduction=16, norm_layer=M.BatchNorm2d,
                 activation=M.ReLU(), attention_act=M.Sigmoid()):
        """

        Args:
            channels (int):
            reduction (int):
            norm_layer (M.Module):
            activation (M.Module):
            attention_act (M.Module):
        """
        super(SEModule, self).__init__()
        inter_ch = int(channels//reduction)
        self.fc = M.Sequential(
            M.AdaptiveAvgPool2d(1),
            Conv2d(channels, inter_ch,norm_layer=norm_layer, activation=activation),
            Conv2d(inter_ch, channels, norm_layer=norm_layer, activation=attention_act)
        )

    def forward(self, x):
        net = self.fc(x)
        return net

class SeparableConv2d(M.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0,
                 bias=False, dilation=1, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.pw_conv = M.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=bias)
        self.dw_conv = M.Conv2d(out_ch, out_ch, kernel_size=ksize, stride=stride, groups=out_ch,
                                padding=padding, dilation=dilation, bias=bias,**kwargs)

    def forward(self, x):
        net = self.pw_conv(x)
        net = self.dw_conv(net)
        return net


class Dropout2d(M.Module):
    def __init__(self, drop_prob, kernel_size, gamma_scale:float=1.0,
                 batch_wise:bool=False, with_noise:bool=False):
        '''
            Implementation of the 2d dropout.
            References:
                <https://arxiv.org/pdf/1810.12890.pdf>
            Args:
                drop_prob (float): the drop out rate
                kernel_size (int): the kernel size of zhe pooling
        '''
        super(Dropout2d, self).__init__()
        self.drop_prob = drop_prob
        self.kernel_size = kernel_size
        self.gama_scale = gamma_scale
        self.batch_wise = batch_wise
        self.with_noise = with_noise

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        _, c, h, w = x.shape
        total_size = w*h
        clipped_block_size = min(self.kernel_size, min(w, h))
        gamma = self.gama_scale*self.drop_prob * total_size / clipped_block_size**2 / (
            (w - self.kernel_size+1) * (h - self.kernel_size + 1)
        )
        if self.batch_wise:
            block_mask = mge.random.uniform(size=(1, h, w, c))
        else:
            block_mask = mge.random.uniform(size=x.shape)
        block_mask[block_mask < gamma] = 1
        block_mask[block_mask >= gamma] = 0
        block_mask = F.max_pool2d(mge.tensor(block_mask, dtype="float32"), kernel_size=int(clipped_block_size),
                                  stride=1, padding=(int(clipped_block_size//2), int(clipped_block_size//2)))
        if self.with_noise:
            normal_noise = mge.random.normal(size=x.shape if not self.batch_wise else (1, c, h, w))
            y = x * (1. - block_mask) + normal_noise*block_mask
        else:
            block_mask = 1 - block_mask
            normalize_scale = (block_mask.size / F.add(block_mask.sum(), 1e-7))
            y = x*block_mask*normalize_scale
        return y

class rSoftmax(M.Module):
    def __init__(self, cardinality, radix):
        """
        Implementation rSoftMax in ResNeSt
        Args:
            cardinality(int): number of cardinality
            radix (int): number of radix
        """
        super(rSoftmax, self).__init__()
        self.cardinality = cardinality
        self.radix = radix

    def forward(self, x):
        bs = x.shape[0]
        if self.radix > 1:
            x = x.reshape(bs, self.cardinality, self.radix,-1).transpose(0, 2, 1, 3)
            x = F.softmax(x, axis=1)
            x = x.reshape(bs, -1)
        else:
            x = F.sigmoid(x)
        return x

class SplAtConv2d(M.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0,
                 radix=2, groups=1, reduction_factor=4, norm_layer=None,
                 dilation=1, dropblock_prob=0.0):
        """
        Implementation of Split Attention Conv in ResNeSt.
        Args:
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output (or number of conv filters)
            ksize (Union[int, tuple]): kernel size of the conv
            stride (Union[int, tuple]): stride of the conv filter
            padding (Union[int, tuple]): padding size of the conv
            radix (int): number of radix
            groups (int): number of groups (number of cardinality)
            reduction_factor (int): reduction rate
            norm_layer (M.Module): module for normalize
            dilation (Union[int, tuple]): dilation rate (ratous)
            dropblock_prob (float): rate of dropout

                           x
                        / ..  \
                       r1 ..   rn
                     /.. \   /.. \
                    c1   cn c1   cn
         """
        super(SplAtConv2d, self).__init__()
        self.radix = radix
        self.cardinality = groups
        self.dropbloc_prob = dropblock_prob
        self.use_bn = norm_layer is not None
        inter_ch = max(int(out_ch*radix//reduction_factor), 32)
        self.conv = M.Conv2d(in_ch, out_ch*radix, kernel_size=ksize, stride=stride, padding=padding,
                             dilation=dilation, groups=groups*radix, bias=False)
        self.relu = M.ReLU()
        if self.use_bn:
            self.bn0 = norm_layer(out_ch*radix)

        self.fc1 = M.Conv2d(out_ch, inter_ch, kernel_size=1, stride=1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_ch)

        self.fc2 = M.Conv2d(inter_ch, out_ch*radix, kernel_size=1, stride=1, groups=self.cardinality)
        self.rsoftmax = rSoftmax(self.cardinality, radix)
        if self.dropbloc_prob > 0.0:
            self.drop_block=Dropout2d(self.dropbloc_prob, 3)

    def forward(self, x):
        net = self.conv(x)
        bs = x.shape[0]
        if self.use_bn:
            net = self.bn0(net)
        net = self.relu(net)
        if self.radix > 1:
            splits = F.split(net, int(self.radix), axis=1)
            gap = sum(splits)
        else:
            gap = net
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)
        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).reshape(bs, -1, 1, 1)

        if self.radix > 1:
            attens = F.split(atten, int(self.radix), axis=1)
            out = sum([att*split for att, split in zip(attens, splits)])
        else:
            out = atten*net
        return out


if __name__ == "__main__":
    x = mge.random.normal(size=(4, 3, 256, 256))
    x = Dropout2d(drop_prob=0.18, kernel_size=7)(x)