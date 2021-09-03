# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  tuple_functools.py
@Time    :  2021/8/30 14:34
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

import collections.abc
import functools


def to_ndtuple(inp, n):
    if isinstance(inp, collections.abc.Iterable):
        assert len(inp) >= n, "The length of input is greater than except length, " \
                             "except {} but got {}".format(n, len(inp))
        return inp[:n]
    else:
        return tuple([int(inp)]*n)


_pair = functools.partial(to_ndtuple,n=2)
