# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  pixelshuffle.py
@Time    :  2021/8/30 15:51
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""


import megengine.module as M

__all__ = ["PixelShuffle"]

class PixelShuffle(M.Module):
    def __init__(self, upscale_factor:int):
        """
        Pixel Shuffle in ESPCN, upsampling method.
        References:
            "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network"
            <https://arxiv.org/abs/1609.05158>
        Args:
            upscale_factor (int): upscale rate
        Examples:
            import numpy as np
            import megengine as mge
            x = np.random.normal(0, 1, (1, 48, 64, 64))
            x = mge.tensor(x)
            up = PixelShuffle(4)
            out = up(x)
        """
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        bs, c, h, w = input.shape
        ow = w * self.upscale_factor
        oh = h * self.upscale_factor
        square = self.upscale_factor**2
        oc = c // square
        assert (oc > 0 and c % square == 0), "pixel_shuffle expects its input's 'channel' dimension to " \
                       "be divisible by the square of upscale_factor, but " \
                       "input.shape[1]={} is not divisible by {}".format(c, square)
        #n, out_ch, scale_rate, scale_rate, input height, input weight
        out = input.reshape(bs, oc, self.upscale_factor, self.upscale_factor, h, w)
        # n, out_ch input height,scale_rate, input weight, scale_rate
        out = out.transpose(0, 1, 4, 3, 5, 2)
        #n, out_ch, output height, output weight
        out = out.reshape(bs, oc, oh, ow)
        return  out

if __name__ == "__main__":
    import megengine as mge
    import numpy as np
    x = mge.tensor(np.random.normal(0, 1, (1, 13, 64, 64)))
    upsample = PixelShuffle(2)
    out = upsample(x)
    print(out.shape)