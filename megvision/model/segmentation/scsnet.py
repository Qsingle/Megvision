# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:scsnet
    author: 12718
    time: 2021/12/9 13:29
    tool: PyCharm
"""

import megengine.module as M
import megengine.functional as F

from megvision.layers import Conv2d
from megvision.layers import AFF, SFA
try:
    from megengine.module import PixelShuffle
except :
    '''
        To compatible with megengine < 1.7.0
    '''
    from megvision.layers import PixelShuffle
from megvision.comm.activation import Tanh
from megvision.layers import FIM


class ResidualBlock(M.Module):
    def __init__(self, in_ch, out_ch, norm_layer=M.BatchNorm2d, activation=M.ReLU()):
        """
            Implementation of the residual block in SCS-Net, we follow the structure
            described in ResNet to build this block.
            References:
                "SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation"
                <https://www.sciencedirect.com/science/article/pii/S1361841521000712#!>
                "Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>

            Parameters:
            ----------
                in_ch
                out_ch
                norm_layer
                activation
        """
        super(ResidualBlock, self).__init__()
        self.conv = M.Sequential(
            Conv2d(in_ch, out_ch, ksize=3, stride=1, padding=1, norm_layer=norm_layer, activation=activation),
            Conv2d(out_ch, out_ch, ksize=3, stride=1, padding=1, norm_layer=norm_layer, activation=None)
        )
        self.activation = activation
        self.identity = M.Identity()
        if in_ch != out_ch:
            self.identity = M.Conv2d(in_ch, out_ch, 1, 1, bias=False)

    def forward(self, x):
        identity = self.identity(x)
        net = self.conv(x)
        net = net + identity
        net = self.activation(net)
        return net


class SCSNet(M.Module):
    def __init__(self, in_ch=3, num_classes=2, super_reso=False, out_size=None, upscale_rate=2, alphas=[0.6, 0.3, 0.1]):
        super(SCSNet, self).__init__()
        base_ch = 64
        self.down = M.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = ResidualBlock(in_ch, base_ch)
        self.encoder2 = ResidualBlock(base_ch, base_ch*2)
        self.encoder3 = ResidualBlock(base_ch*2, base_ch*4)

        self.sfa = SFA(base_ch*4)

        self.aff3 = AFF(base_ch*4)
        self.aff2 = AFF(base_ch*2)
        self.aff_conv3 = M.Sequential(
            M.Conv2d(base_ch*4, base_ch*2, 3, stride=1, padding=1, bias=False),
            M.ReLU()
        )
        self.aff1 = AFF(base_ch)
        self.aff_conv2 = M.Sequential(
            M.Conv2d(base_ch*2, base_ch, 3, stride=1, padding=1, bias=False),
            M.ReLU()
        )
        self.aff_conv1 =  M.Sequential(
            M.Conv2d(base_ch, base_ch, 3, stride=1, padding=1, bias=False),
            M.ReLU()
        )

        self.side_l3 = M.Conv2d(base_ch*2, num_classes, 1)
        self.side_l2 = M.Conv2d(base_ch, num_classes, 1)
        self.side_l1 = M.Conv2d(base_ch, num_classes, 1)

        self.alpha_l3 = alphas[2]
        self.alpha_l2 = alphas[1]
        self.alpha_l1 = alphas[0]
        self.upscale_rate = upscale_rate
        self.super_reso = super_reso
        self.out_size = out_size
        if super_reso:
            self.sr_aff3 = AFF(base_ch*4)
            self.sr_aff3_conv = M.Sequential(
                M.Conv2d(base_ch*4, base_ch*2, 3, 1, padding=1, bias=False),
                M.ReLU()
            )

            self.sr_aff2 = AFF(base_ch * 2)
            self.sr_aff2_conv = M.Sequential(
                M.Conv2d(base_ch * 2, base_ch, 3, 1, padding=1, bias=False),
                M.ReLU()
            )

            self.sr_aff1 = AFF(base_ch)
            self.sr_aff1_conv = M.Sequential(
                M.Conv2d(base_ch, base_ch, 3, 1, padding=1, bias=False),
                M.ReLU()
            )

            self.sr = M.Sequential(
                M.Conv2d(base_ch, 64, kernel_size=5, stride=1, padding=2, bias=False),
                Tanh(),
                M.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                Tanh(),
                M.Conv2d(32, (upscale_rate ** 2) * in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                PixelShuffle(upscale_factor=upscale_rate)
            )
            self.query = FIM(in_ch, num_classes, hidden_state=16)

    def forward(self, x):
        en1 = self.encoder1(x)
        down1 = self.down(en1)
        en2 = self.encoder2(down1)
        down2 = self.down(en2)
        en3 = self.encoder3(down2)
        down3 = self.down(en3)
        sfa = self.sfa(down3)
        sfa = F.vision.interpolate(sfa, size=en3.shape[2:], mode="bilinear", align_corners=True)
        aff3 = self.aff3(en3, sfa)
        aff3 = self.aff_conv3(aff3)
        aff3_up = F.vision.interpolate(aff3, size=en2.shape[2:], mode="bilinear", align_corners=True)
        aff2 = self.aff2(en2, aff3_up)
        aff2 = self.aff_conv2(aff2)
        aff2_up = F.vision.interpolate(aff2, size=en1.shape[2:], mode="bilinear", align_corners=True)
        aff1 = self.aff1(en1, aff2_up)
        aff1 = self.aff_conv1(aff1)
        side1 = self.side_l1(aff1)
        side2 = self.side_l2(F.vision.interpolate(aff2, size=x.shape[2:], mode="bilinear",
                                                  align_corners=True))
        side3 = self.side_l3(F.vision.interpolate(aff3, size=x.shape[2:], mode="bilinear",
                                                  align_corners=True))
        out = self.alpha_l1 * side1 + self.alpha_l2*side2 + self.alpha_l3*side3
        if self.super_reso and self.out_size is not None:
            out = F.vision.interpolate(out, size=self.out_size, mode="bilinear",
                                       align_corners=True)
        sr = None
        qr_seg = None
        if self.super_reso:
            out = F.vision.interpolate(out, scale_factor=self.upscale_rate, mode="bilinear", align_corners=True)
            if self.training:
                sr_aff3 = self.sr_aff3(en3, sfa)
                sr_aff3 = self.sr_aff3_conv(sr_aff3)
                sr_aff3_up = F.vision.interpolate(sr_aff3, size=en2.shape[2:], mode="bilinear", align_corners=True)
                sr_aff2 = self.sr_aff2(en2, sr_aff3_up)
                sr_aff2 = self.sr_aff2_conv(sr_aff2)
                sr_aff2_up = F.vision.interpolate(sr_aff2, size=en1.shape[2:], mode="bilinear", align_corners=True)
                sr_aff1 = self.sr_aff1(en1, sr_aff2_up)
                aff1 = self.sr_aff1_conv(sr_aff1)
                sr = self.sr(aff1)
                if self.out_size is not None:
                    sr = F.vision.interpolate(sr, size=self.out_size, mode="bilinear",
                                       align_corners=True)
        if self.super_reso and self.training:
            qr_seg, qr_sr = self.query(sr, out)
        if sr is None:
            return out
        return out, qr_sr, qr_seg
