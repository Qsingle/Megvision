# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  poly_lr_scheduler.py
@Time    :  2021/8/31 16:24
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

from .lr_scheduler import LRScheduler

class PolyLRScheduler(LRScheduler):
    def __init__(self, optimizer,  num_images, batch_size, epochs, gamma=0.9, start=-1):
        super(PolyLRScheduler, self).__init__(optimizer, start)
        self.total_iterations = (num_images) // batch_size * epochs
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_images = num_images
        self.epochs = epochs

    def get_lr(self):
        return [group["initia_lr"] * ((1 - self.current_step /
                                       self.total_iterations)**self.gamma) for group in
                self.optimizer.param_groups]

    def state_dict(self):
        return {
            key:value
            for key,value in self.__dict__.items()
            if key in ["total_iterations", "gamma", "current_step",
                       "batch_size", "num_images", "epochs"]
        }

    def load_state_dict(self, state_dict):
        tmp_state = {}
        keys = ["total_iterations", "gamma", "current_step",
                       "batch_size", "num_images", "epochs"]
        for key in keys:
            if key not in state_dict:
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_state[key] = state_dict[key]
        self.__dict__.update(tmp_state)