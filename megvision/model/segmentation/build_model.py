# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  build_model.py
@Time    :  2021/8/31 10:33
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

from megvision.comm.register import Register

SEGMENTATION_REGISTER = Register("Segmentation")

def create_model(name):
    return SEGMENTATION_REGISTER.get(name)