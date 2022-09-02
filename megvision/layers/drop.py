# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:drop
    author: 12718
    time: 2022/1/12 15:10
    tool: PyCharm
"""
import megengine as mge

def drop_path(x:mge.Tensor, drop_prob:float, training:bool, scale_by_keep:bool=True):
    """
    Implementation of the dropout path
    Parameters
    ----------
    x (mge.Tensor): input tensor
    drop_prob (float): dropout probability
    training (bool): whether training mode
    scale_by_keep (bool) whether scaled by the keep probability
    References:
        https://github.com/rwightman/pytorch-image-models/blob/6c17d57a2c4c94fb0a1b6a6f66a64bf4e0264400/timm/models/layers/drop.py
    Returns
    -------
        Tensor
    """
    if drop_prob == 0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,)*(x.ndim -1)
    random_tensor = keep_prob + mge.tensor(mge.random.uniform(0, 1, size=shape), dtype=x.dtype, device=x.device)
    random_tensor = mge.functional.floor(random_tensor) #binarized
    if scale_by_keep:
        random_tensor = random_tensor / keep_prob
    out = x * random_tensor
    return out