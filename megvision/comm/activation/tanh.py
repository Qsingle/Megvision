# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:tanh
    author: 12718
    time: 2021/11/17 17:04
    tool: PyCharm
"""

import megengine.module as M
import megengine.functional as F

class Tanh(M.Module):
    def forward(self, inputs):
        return F.tanh(inputs)