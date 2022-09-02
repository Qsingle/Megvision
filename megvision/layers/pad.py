# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:pad
    author: 12718
    time: 2021/12/14 15:27
    tool: PyCharm
"""
import megengine.functional as F
import math
from typing import List, Tuple

'''
References:
'''

def get_same_padding(x:int, k:int, s:int, d:int):
    """
    calculate the size to pad
    Parameters
    ----------
    x (int): input size
    k (int): kernel size
    s (int): stride
    d (int): dilation

    Returns
    -------
       int: size to pad
    """
    return max((math.ceil(x / s) - 1)*s + (k-1)*d + 1 -x, 0)

def pad_same(x, k:List[int], s:List[int], d:List[int], value:float=0):
    """
    Padding the input tensor as same padding in TensorFlow, dynamic mode
    Parameters
    ----------
    x (Tensor): input tensor
    k (List[int]): kernel size of conv
    s (List[int]): stride of conv
    d (List[int]): dilation of conv
    value (float): value

    Returns
    -------
        tensor after pad
    """
    ih, iw = x.shape[2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.nn.pad(x, ((0, 0), (0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w//2, pad_w - pad_w // 2)), constant_value=value)
    return x