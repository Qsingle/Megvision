# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  __init__.py.py
@Time    :  2021/8/26 16:27
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

from .deeplab import *
from .unet import *
from .espnet import ESPNetV2_Seg
from .pspnet import PSPNet
from .cs2net import CSNet
from .cenet import CENet
from .scsnet import SCSNet

from .build_model import create_model