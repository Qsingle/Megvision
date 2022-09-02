# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:ema
    author: 12718
    time: 2021/9/23 9:39
    tool: PyCharm
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy

import megengine.module as M
from megengine.functional.inplace import _inplace_add_

class ModelEMA(M.Module):
    def __init__(self, model:M.Module, alpha=0.999):
        super(ModelEMA, self).__init__()
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def _update(self, model, update_fn):
        for ema_v, model_v in zip(self.model.parameters(), model.parameters()):
            value = update_fn(ema_v.detach(), model_v.detach())
            ema_v.set_value(value)

    def forward(self, model):
        self._update(model, update_fn=lambda e, m: self.alpha*e + (1-self.alpha)*m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)