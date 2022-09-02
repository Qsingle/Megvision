# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  unet.py
@Time    :  2021/9/1 15:19
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import megengine as mge
import megengine.module as M
import megengine.functional as F

from megvision.layers.layers import Conv2d
try:
    from megengine.module import PixelShuffle
except :
    '''
        To compatible with megengine < 1.7.0
    '''
    from megvision.layers import PixelShuffle
from megvision.comm.activation import Tanh
from megvision.layers.sa import SA
from megvision.layers import FIM

from .build_model import SEGMENTATION_REGISTER

__all__ = ["UNet", "SAUNet"]

class DoubleConv(M.Module):
    def __init__(self, in_ch, out_ch, expansion=1.0,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU(),
                 dropout=0.0, drop_block_size=0):
        super(DoubleConv, self).__init__()
        inter_ch = max(int(out_ch*expansion), 16)
        self.conv = M.Sequential(
            Conv2d(in_ch, inter_ch, ksize=3, stride=1, padding=1,
                   norm_layer=norm_layer, activation=activation,
                   dropout=dropout, drop_block_size=drop_block_size),
            Conv2d(inter_ch, out_ch, ksize=3, stride=1, padding=1,
                   norm_layer=norm_layer, activation=activation,
                   dropout=dropout, drop_block_size=drop_block_size)
        )

    def forward(self, x):
        net = self.conv(x)
        return net

class Downsample(M.Module):
    def __init__(self, in_ch, out_ch, expansion=1.0,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU(),
                 dropout=0.0, drop_block_size=0):
        super(Downsample, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch, expansion=expansion,
                               norm_layer=norm_layer, activation=activation,
                               dropout=dropout, drop_block_size=drop_block_size)
        self.down = M.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        down_pre = self.conv(x)
        down = self.down(down_pre)
        return down_pre, down
    

class Upsample(M.Module):
    def __init__(self, in_ch1, in_ch2, out_ch,
                 expansion=1.0, norm_layer=M.BatchNorm2d,
                 activation=M.ReLU(),
                 dropout=0.0, drop_block_size=0,
                 bilinear=True
                 ):
        super(Upsample, self).__init__()
        self.conv = DoubleConv(in_ch2*2, out_ch, expansion=expansion,
                               norm_layer=norm_layer,
                               activation=activation,
                               dropout=dropout,
                               drop_block_size=drop_block_size)
        self.bilinear = bilinear
        if not bilinear:
            self.upsample = M.ConvTranspose2d(in_ch1, in_ch2, kernel_size=3, stride=2, padding=1)
        else:
            self.upsamle_conv = Conv2d(in_ch1, in_ch2)


    def forward(self, x1, x2):
        if self.bilinear:
            upsample = F.vision.interpolate(x1, size=x2.shape[2:], align_corners=True)
            upsample = self.upsamle_conv(upsample)
        else:
            upsample = self.upsample(x1)
            upsample = F.nn.pad(upsample, ((0, 0), (0, 0), (1, 0), (1,0)))
        cat = F.concat([upsample, x2], axis=1)
        out = self.conv(cat)
        return out

@SEGMENTATION_REGISTER.register()
class SAUNet(M.Module):
    def __init__(self, in_ch, num_classes, block_size=7, keep_prob=0.82, super_reso=False,
                 expansion=1.0, upscale_rate=4, scale=1, norm_layer=M.BatchNorm2d, activation=M.ReLU(),
                 output_size=None, bilinear=True):
        super(SAUNet, self).__init__()
        drop_prob = 1 - keep_prob
        base_ch = int(16*scale)
        self.down1 = Downsample(in_ch, base_ch, expansion=expansion,
                                norm_layer=norm_layer,
                                activation=activation,
                                dropout=drop_prob, drop_block_size=block_size)
        self.down2 = Downsample(base_ch, base_ch*2, expansion=expansion,
                                norm_layer=norm_layer,
                                activation=activation,
                                dropout=drop_prob, drop_block_size=block_size)
        self.down3 = Downsample(base_ch*2, base_ch*4, expansion=expansion,
                                norm_layer=norm_layer,
                                activation=activation,
                                dropout=drop_prob, drop_block_size=block_size)
        self.down4 = M.Sequential(
            Conv2d(base_ch*4, base_ch*8, 3, padding=1,
                   dropout=drop_prob,
                   drop_block_size=block_size,
                   activation=activation,
                   norm_layer=norm_layer),
            SA(),
            Conv2d(base_ch*8, base_ch*8, 3, padding=1,
                   dropout=drop_prob,
                   drop_block_size=block_size,
                   activation=activation,
                   norm_layer=norm_layer)
        )

        self.up6 = Upsample(base_ch*8, base_ch*4, base_ch*4, expansion=expansion,
                            norm_layer=norm_layer,
                            activation=activation,
                            dropout=drop_prob,
                            drop_block_size=block_size,
                            bilinear=bilinear
                            )
        self.up7 = Upsample(base_ch*4, base_ch*2, base_ch*2, expansion=expansion,
                            norm_layer=norm_layer,
                            activation=activation,
                            dropout=drop_prob,
                            drop_block_size=block_size,
                            bilinear=bilinear
                            )
        self.up8 = Upsample(base_ch*2, base_ch, base_ch, expansion=expansion,
                            norm_layer=norm_layer,
                            activation=activation,
                            dropout=drop_prob,
                            drop_block_size=block_size,
                            bilinear=bilinear
                            )
        self.out_conv = M.Conv2d(16, num_classes, kernel_size=1, stride=1)

        self.super_reso = super_reso
        if super_reso:
            self.sr_up6 = Upsample(base_ch*8, base_ch*4, base_ch*4,expansion=expansion,
                                   norm_layer=norm_layer,
                                   activation=activation,
                                   dropout=drop_prob,
                                   drop_block_size=block_size,
                                   bilinear=bilinear
                                   )
            self.sr_up7 = Upsample(base_ch*4, base_ch*2, base_ch*2, expansion=expansion,
                                   norm_layer=norm_layer,
                                   activation=activation,
                                   dropout=drop_prob,
                                   drop_block_size=block_size,
                                   bilinear=bilinear
                                   )
            self.sr_up8 = Upsample(base_ch*2, base_ch, base_ch, expansion=expansion,
                                   norm_layer=norm_layer,
                                   activation=activation,
                                   dropout=drop_prob,
                                   drop_block_size=block_size,
                                   bilinear=bilinear
                                   )

            self.sup = M.Sequential(
                M.Conv2d(32, (upscale_rate ** 2) * in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                PixelShuffle(upscale_factor=upscale_rate)
            )
            self.scale_factor = upscale_rate
            self.sup_conv = M.Sequential(
                M.Conv2d(base_ch, 64, kernel_size=5, stride=1, padding=2, bias=False),
                Tanh(),
                M.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                Tanh()
            )
            # self.fa = M.Sequential(
            #     M.Conv2d(num_classes, in_ch, 1, 1, 0, bias=False),
            #     M.BatchNorm2d(in_ch),
            #     M.ReLU()
            # )
            self.interaction = FIM(in_ch, num_classes)
            self.output_size = output_size

    def forward(self, x):
        down1_0, down1 = self.down1(x)
        down2_0, down2 = self.down2(down1)
        down3_0, down3 = self.down3(down2)
        down4 = self.down4(down3)
        hr = None
        up6 = self.up6(down4, down3_0)
        up7 = self.up7(up6, down2_0)
        up8 = self.up8(up7, down1_0)
        if self.super_reso:
            if self.training:
                hr_net = self.sr_up6(down4, down3_0)
                hr_net = self.sr_up7(hr_net, down2_0)
                hr_net = self.sr_up8(hr_net, down1_0)
                hr_fe = self.sup_conv(hr_net)
                hr = self.sup(hr_fe)
                if self.output_size is not None:
                    h, w = hr.shape[2:]
                    if h != self.output_size[0] or w != self.output_size[1]:
                        hr = F.vision.interpolate(hr, size=self.output_size, mode="bilinear", align_corners=True)

                    # up9 = self.hr_down(hr) + up9
        out = self.out_conv(up8)
        if self.super_reso:
            out = F.vision.interpolate(out, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
            if self.output_size is not None:
                h, w = out.shape[2:]
                if h != self.output_size[0] or w != self.output_size[1]:
                    out = F.vision.interpolate(out, size=self.output_size, mode="bilinear", align_corners=True)
            if self.training:
                qr_out = self.interaction(hr, out)
                # fa = self.fa(qr_out)
        if hr is not None:
            return out, qr_out
        return out


@SEGMENTATION_REGISTER.register()
class UNet(M.Module):
    def __init__(self, in_ch, num_classes, expansion=1.0,
                 norm_layer=M.BatchNorm2d, activation=M.ReLU(),
                 super_reso=False, upscale_rate=4, output_size=None,
                 sssr=False):
        """
        Implementation Unet
        References:
        "U-Net: Convolutional Networks for Biomedical Image Segmentation"<http://www.arxiv.org/pdf/1505.04597.pdf>

        Parameters
        ----------
        in_ch (int): number of channels of input
        num_classes (int): number of classes
        expansion (float): expansion rate
        norm_layer (M.Module): normalization module
        activation (M.Module object): activation function
        super_reso (bool): whether use super resolution
        upscale_rate (int): upscale rate for the super resolution
        """
        super(UNet, self).__init__()
        self.down1 = Downsample(in_ch, 64, expansion=expansion,
                                norm_layer=norm_layer,
                                activation=activation)
        self.down2 = Downsample(64, 128, expansion=expansion,
                                norm_layer=norm_layer,
                                activation=activation)
        self.down3 = Downsample(128, 256, expansion=expansion,
                                norm_layer=norm_layer,
                                activation=activation)
        self.down4 = Downsample(256, 512, expansion=expansion,
                                norm_layer=norm_layer,
                                activation=activation)
        self.down5 = DoubleConv(512, 1024, expansion=expansion,
                                norm_layer=norm_layer,
                                activation=activation)
        self.up6 = Upsample(1024, 512, 512, expansion=expansion,
                            norm_layer=norm_layer, activation=activation)
        self.up7 = Upsample(512, 256, 256, expansion=expansion,
                            norm_layer=norm_layer, activation=activation)
        self.up8 = Upsample(256, 128, 128, expansion=expansion,
                            norm_layer=norm_layer, activation=activation)
        self.up9 = Upsample(128, 64, 64, expansion=expansion,
                            norm_layer=norm_layer, activation=activation)
        self.out_conv = M.Conv2d(64, num_classes, kernel_size=1, stride=1)

        self.super_reso = super_reso
        self.sssr = sssr
        if super_reso:
            self.sr_up6 = Upsample(1024, 512, 512, activation=activation,
                                   norm_layer=norm_layer, expansion=expansion)
            self.sr_up7 = Upsample(512, 256, 256, norm_layer=norm_layer, activation=activation,
                                   expansion=expansion)
            self.sr_up8 = Upsample(256, 128, 128, expansion=expansion, norm_layer=norm_layer,
                                   activation=activation)
            self.sr_up9 = Upsample(128, 64, 64, expansion=expansion,
                                   norm_layer=norm_layer, activation=activation)

            self.sup = M.Sequential(
                M.Conv2d(32, (upscale_rate**2)*in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                PixelShuffle(upscale_factor=upscale_rate)
            )
            self.scale_factor = upscale_rate
            self.sup_conv = M.Sequential(
                M.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=False),
                Tanh(),
                M.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                Tanh()
            )
            # self.fa = M.Sequential(
            #     M.Conv2d(num_classes, in_ch, 1, 1, 0, bias=False),
            #     M.BatchNorm2d(in_ch),
            #     M.ReLU()
            # )
            self.interaction = FIM(in_ch, num_classes, hidden_state=16)
            self.output_size = output_size

    def forward(self, x):
        down1_0, down1 = self.down1(x)
        down2_0, down2 = self.down2(down1)
        down3_0, down3 = self.down3(down2)
        down4_0, down4 = self.down4(down3)
        hr = None
        down5 = self.down5(down4)
        up6 = self.up6(down5, down4_0)
        up7 = self.up7(up6, down3_0)
        up8 = self.up8(up7, down2_0)
        up9 = self.up9(up8, down1_0)
        if self.super_reso:
            if self.training:
                hr_up6 = self.sr_up6(down5, down4_0)
                hr_up7 = self.sr_up7(hr_up6, down3_0)
                hr_up8 = self.sr_up8(hr_up7, down2_0)
                hr_up9 = self.sr_up9(hr_up8, down1_0)
                hr_fe = self.sup_conv(hr_up9)
                # up9, hr_fe = self.query_module(hr_fe, up9)
                hr = self.sup(hr_fe)
                # up9 = self.hr_down(hr) + up9
                if self.output_size is not None:
                    h, w = hr.shape[2:]
                    if h != self.output_size[0] or w != self.output_size[1]:
                        hr = F.vision.interpolate(hr, size=self.output_size, mode="bilinear", align_corners=True)
        out = self.out_conv(up9)
        # if self.sssr or self.super_reso:
        #     out = F.vision.interpolate(out, scale_factor=2, mode="bilinear", align_corners=True)
        if self.super_reso:
            # out = F.vision.interpolate(out, scale_factor=self.scale_factor, mode="bilinear", align_corners=True)
            if self.output_size is not None:
                h, w = out.shape[2:]
                if h != self.output_size[0] or w != self.output_size[1]:
                    out = F.vision.interpolate(out, size=self.output_size, mode="bilinear", align_corners=True)
            out_seg = None
            if self.training:
                out_seg = self.interaction(hr, out)

        if hr is not None:
            return out, hr, out_seg
        return out

if __name__ == "__main__":
    import numpy as np
    import megengine as mge
    x = mge.tensor(np.random.normal(0, 1, size=(4, 3, 512, 512)))
    model = UNet(3, 3, super_reso=True, upscale_rate=2)
    out1, out2 = model(x)
    print(out2.shape)