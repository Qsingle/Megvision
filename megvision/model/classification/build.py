# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  build.py
@Time    :  2021/8/26 16:20
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

from comm.register import Register

BACKBONE_REGISTER = Register("BACKBONE")

def get_model(name):
    return BACKBONE_REGISTER.get(name)
