# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:hsigmoid
    author: 12718
    time: 2022/1/13 8:00
    tool: PyCharm
"""

import megengine.module as M
import megengine.functional as F


try:
    from megengine.functional import hsigmoid
except:
    def hsigmoid(x):
        return F.relu6(x + 3) / 6


class HSigmoid(M.Module):
    def __init__(self):
        super(HSigmoid, self).__init__()

    def forward(self, x):
        return hsigmoid(x)

if __name__ == "__main__":
    import megengine as mge
    x = mge.random.normal(size=(1, 3, 1024, 1024))
    out = HSigmoid()(x)
