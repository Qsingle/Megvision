# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  segmentation_metrics.py
@Time    :  2021/8/30 19:57
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

import numpy as np
from .confusion_matrix import compute_confusion_matrix_numpy

class SegmentationMetrcNumpy:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        num_classes = 2 if self.num_classes == 1 else self.num_classes #only support num_classes > 2
        self.matrix = np.zeros(shape=(num_classes, num_classes))

    def add_batch(self, output, target):
        output = np.array(output)
        target = np.array(target)
        num_classes = self.num_classes if self.num_classes != 1 else 2
        if self.num_classes == 1:
            output = np.where(output >= 0.5, 1, 0)
        if output.ndim == 4:
            output = np.argmax(output, axis=1)
        # for i in range(len(output)):
        self.matrix += compute_confusion_matrix_numpy(output, target, num_classes)


    def evaluate(self):
        TP = np.diag(self.matrix)
        FP = np.sum(self.matrix, axis=0) - TP
        FN = np.sum(self.matrix, axis=1) - TP
        TN = np.sum(self.matrix) - TP - FN - FP
        ACC = (TP + TN) / (TP + FP + TN + FN + 1e-9)
        P = TP / (TP + FP + 1e-9)
        R = TP / (TP + FN + 1e-9)
        SP = TN / (TN + FP + 1e-9)
        IoU = TP / (TP + FN + FP + 1e-9)
        Dice = 2*TP / (2*TP+FN+FP + 1e-9)
        GMean = np.sqrt(R*SP)
        mean_acc = np.nanmean(ACC)
        mAP = np.nanmean(P)
        mAR = np.nanmean(R)
        mSP = np.nanmean(SP)
        mIoU = np.nanmean(IoU)
        mDice = np.nanmean(Dice)
        results = {}
        results["IoU"] = IoU
        results["dice"] = Dice
        results["P"] = P
        results["R"] = R
        results["SP"] = SP
        results["ACC"] = ACC
        results["mean_acc"] = mean_acc
        results["mean_p"] = mAP
        results["mean_r"] = mAR
        results["mean_sp"] = mSP
        results["miou"] = mIoU
        results["mdice"] = mDice
        results["gmean"] = GMean
        return results