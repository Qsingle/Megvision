# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:random_mask
    author: 12718
    time: 2021/11/16 15:44
    tool: PyCharm
"""
import random
import math
import numpy as np





if __name__ == "__main__":
    import cv2
    import megengine as mge
    img = cv2.imread("D:/workspace/datasets/imagenet/ILSVRC2012_img_val/ILSVRC2012_val_00001201.JPEG")
    img = cv2.resize(img, (256, 256))
