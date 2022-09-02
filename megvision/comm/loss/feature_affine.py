# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:feature_affine
    author: 12718
    time: 2021/11/14 13:32
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import megengine as mge
import megengine.module as M
import megengine.functional as F

class FALoss(M.Module):
    def __init__(self, downsample_rate=16):
        """
        Implementation of feature affine loss introduced in
        "Dual Super-Resolution Learning for Semantic Segmentation"
        <https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Dual_Super-Resolution_Learning_for_Semantic_Segmentation_CVPR_2020_paper.html>
        Parameters:
            downsample_rate(int): scale factor of the features, default is 16. In paper is 8
        Returns:
            None
        """
        super(FALoss, self).__init__()
        self.downsample_rate = downsample_rate

    def forward(self, f1, f2):
        """
        Calculate the loss
        Parameters
        ----------
        f1(Tensor): feature of the super resolution head
        f2(Tensor): feature of the segmentation head

        Returns
        -------
            loss scale
        """
        f1 = F.avg_pool2d(f1, kernel_size=self.downsample_rate)
        f2 = F.avg_pool2d(f2, kernel_size=self.downsample_rate)
        assert f1.size == f2.size, "Size of feature1 and feature2 must be equal, but got {} and {}".format(f1.size, f2.size)
        bs, ch, h, w = f1.shape
        f1 = F.reshape(f1, (bs, ch, -1)) #bs, C, h*w
        f2 = F.reshape(f2, (bs, ch, -1)) #bs, C, h*w
        # f1 = f1 / F.norm(f1, 2, axis=1, keepdims=True)
        # f2 = f2 / F.norm(f2, 2, axis=1, keepdims=True)
        mat1 = F.matmul(f1.transpose(0, 2, 1), f1) #bs, h*w, h*w
        mat2 = F.matmul(f2.transpose(0, 2, 1), f2) #bs, h*w, h*w
        loss = F.norm(mat2-mat1, 1, axis=1)
        return loss.mean()


if __name__ == "__main__":
    f1 = mge.random.normal(size=(1, 3, 1024, 1024))
    f2 = mge.random.normal(size=(1, 3, 1024, 1024))
    print(FALoss()(f1, f2))