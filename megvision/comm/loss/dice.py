# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:dice
    author: 12718
    time: 2021/12/7 19:27
    tool: PyCharm
"""
import megengine.functional as F
import megengine.module as M

def dice_loss(output, label, smooth=1.0):
    """
    Calculate the dice loss

    Parameters
    ----------
    output (Tensor): the output of the model
    label (Tensor): the ground truth

    Returns
    -------
        scale-value of dice coefficient
    """
    ns = output.shape[1]
    if ns > 1:
        output = F.softmax(output, axis=1)
        label_onehot = F.one_hot(label, num_classes=ns)
        label_onehot = F.transpose(label_onehot, (0, 3, 1, 2))
        axis = [1, 2, 3]
        inter = F.sum(output * label_onehot, axis=axis)
        union = F.sum(output + label_onehot, axis=axis)
        loss = F.mean(1 - (2*inter + smooth) / (union + smooth))
    else:
        output = F.sigmoid(output)
        inter = F.sum(output * label)
        union = F.sum(output + label)
        dice = (2*inter+smooth) / (union+smooth)
        loss = 1 - dice
    return loss

class DiceLoss(M.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, label):
        return dice_loss(output, label, smooth=self.smooth)