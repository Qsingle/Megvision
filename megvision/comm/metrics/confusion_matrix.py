# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  confusion_matrix.py
@Time    :  2021/8/30 19:45
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

def compute_confusion_matrix_numpy(output, target,num_classes=None):
    """
    Computer the confusion matrix
    Args:
        output (np.array): the output array
        target (np.array): the target array
        num_classes (int): number of classes

    Returns:
        Confusion matrix
    """
    if num_classes is None:
        min_length = np.max(target)
    else:
        if isinstance(num_classes, int):
            min_length = num_classes
        else:
            raise ValueError("Except type of num_classes is int, but got {}".format(type(num_classes)))
    target_mask = (target >= 0) & (target < min_length)
    label = target[target_mask]
    output = output[target_mask]
    pred = label * min_length + output
    matrix = np.bincount(pred, minlength=min_length**2)
    return matrix.reshape((min_length, min_length))