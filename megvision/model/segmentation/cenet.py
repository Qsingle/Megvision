# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:cenet
    author: 12718
    time: 2021/12/8 16:46
    tool: PyCharm
"""
import megengine.module as M
import megengine.functional as F

from megvision.layers import RMP, DAC, Conv2d
from megvision.comm.activation import Tanh
from megvision.model.classification.resnet import resnet34
from megvision.layers import FIM

try:
    from megengine.module import PixelShuffle
except :
    from megvision.layers import PixelShuffle


from .build_model import SEGMENTATION_REGISTER

class ResidualDecoder(M.Module):
    def __init__(self, in_ch, out_ch):
        """
            Implementation of decoder of CENet
            References:
                "CE-Net: Context Encoder Network for 2D Medical Image Segmentation"
                <https://arxiv.org/pdf/1903.02740.pdf>
            Parameters
            ----------
            in_ch (int) number of input channels
        """
        super(ResidualDecoder, self).__init__()
        hidden_ch = int(in_ch//4)
        self.conv1 = Conv2d(in_ch, hidden_ch)

        self.deconv2 = M.Sequential(
            M.ConvTranspose2d(hidden_ch, hidden_ch, 3, stride=2, padding=1),
            M.Pad(((0, 0), (0, 0), (0, 1), (0, 1))),
            M.BatchNorm2d(hidden_ch),
            M.ReLU()
        )
        self.conv3 = Conv2d(hidden_ch, out_ch)

    def forward(self, x):
        net = self.conv1(x)
        net = self.deconv2(net)
        net = self.conv3(net)
        return net

class DACV2(M.Module):
    def __init__(self, in_ch):
        super(DACV2, self).__init__()
        self.dilate1_1 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dilate1_2 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=3, dilation=3)
        self.dilate1_3 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=5, dilation=5)
        self.conv1_1x1 = M.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.dilate2_1 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dilate2_2 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv2_1x1 = M.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.dilate3_1 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv3_1x1 = M.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.dilate4 = M.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        self.activation = M.ReLU()

    def forward(self, x):
        branch1 = self.conv1_1x1(self.dilate1_3(self.dilate1_2(self.dilate1_1(x))))
        branch2 = self.conv2_1x1(self.dilate2_2(self.dilate2_1(x)))
        branch3 = self.conv3_1x1(self.dilate3_1(x))
        branch4 = self.dilate4(x)
        net = x + branch1 + branch2 + branch3 + branch4
        net = self.activation(net)
        return net

@SEGMENTATION_REGISTER.register()
class CENet(M.Module):
    def __init__(self, in_ch, num_classes,
                 super_reso=False, upscale_rate=2, out_size=None):
        super(CENet, self).__init__()
        self.backbone = resnet34(pretrained=True)
        if in_ch != 3:
            inplanes = self.backbone.conv1.out_channels
            del self.backbone.conv1
            self.backbone.conv1 = M.Conv2d(in_ch, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        del self.backbone.fc
        del self.backbone.avgpool

        # self.dac = DAC(512)
        self.dac = DAC(512)
        self.rmp = RMP(512)

        self.decoder4 = ResidualDecoder(516, 256)
        self.decoder3 = ResidualDecoder(256, 128)
        self.decoder2 = ResidualDecoder(128, 64)
        self.decoder1 = ResidualDecoder(64, 64)

        self.finaldeconv1 = M.Sequential(
            M.ConvTranspose2d(64, 32, 4, 2),
            M.ReLU()
        )
        self.finalconv2 = M.Sequential(
            M.Conv2d(32, 32, 3, 1, 1),
            M.ReLU()
        )
        self.finalconv3 = M.Conv2d(32, num_classes, 3, 1)

        self.super_reso = super_reso
        self.out_size = out_size
        self.upscale_rate = upscale_rate

        if self.super_reso:
            self.sr_de4 = ResidualDecoder(516, 256)
            self.sr_de3 = ResidualDecoder(256, 128)
            self.sr_de2 = ResidualDecoder(128, 64)
            self.sr_de1 = ResidualDecoder(64, 64)
            self.sr_final = M.Sequential(
                M.ConvTranspose2d(64, 32, 4, 2),
                M.ReLU(),
                M.Conv2d(32, 32, 3, 1, 1),
                M.BatchNorm2d(32),
                M.ReLU(),
                M.Conv2d(32, 32, 3, 1),
                M.BatchNorm2d(32),
                M.ReLU()
            )
            # self.sr_de4 = Upsample(516, 256, 256)
            # self.sr_de3 = Upsample(256, 128, 128)
            # self.sr_de2 = Upsample(128, 64, 64)
            # self.sr_de1 = M.Sequential(
            #     Conv2d(128, 64, 3, 1, 1),
            #     Conv2d(64, 64, 3, 1, 1),
            #     Conv2d(64, 32, 1, 1)
            # )
            self.sr = M.Sequential(
                M.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
                Tanh(),
                M.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
                Tanh(),
                M.Conv2d(32, (upscale_rate ** 2) * in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                PixelShuffle(upscale_factor=upscale_rate)
            )
            self.interaction = FIM(in_ch, num_classes)

    def forward(self, x):
        f1 = self.backbone.conv1(x)
        f1 = self.backbone.bn1(f1)
        f1_0 = self.backbone.relu(f1)
        f1 = self.backbone.maxpool(f1_0)

        e1 = self.backbone.layer1(f1)
        e2 = self.backbone.layer2(e1)
        e3 = self.backbone.layer3(e2)
        e4_0 = self.backbone.layer4(e3)


        e4 = self.dac(e4_0)

        e4 = self.rmp(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalconv2(out)
        out = self.finalconv3(out)
        sr = None
        if self.super_reso:
            out = F.vision.interpolate(out, scale_factor=self.upscale_rate, align_corners=True)
            if self.training:
                sr_d4 = self.sr_de4(e4) + e3
                sr_d3 = self.sr_de3(sr_d4) + e2
                sr_d2 = self.sr_de2(sr_d3) + e1
                sr_d1 = self.sr_de1(sr_d2)
                sr = self.sr_final(sr_d1)
                # sr_d4 = self.sr_de4(e4, e3)
                # sr_d3 = self.sr_de3(sr_d4, e2)
                # sr_d2 = self.sr_de2(sr_d3, e1)
                # sr_d2 = F.vision.interpolate(sr_d2, size=f1_0.shape[2:], mode="bilinear", align_corners=True)
                # sr = self.sr_de1(F.concat([f1_0, sr_d2], axis=1))
                # sr = F.vision.interpolate(sr, size=x.shape[2:], mode="bilinear", align_corners=True)
                sr = self.sr(sr)

        if self.out_size is not None and self.super_reso:
            out = F.vision.interpolate(out, size=self.out_size, align_corners=True)
            if self.training:
                sr = F.vision.interpolate(sr, size=self.out_size, align_corners=True)
        if self.super_reso and self.training:
            qr_seg = self.interaction(sr, out)
        if sr is not None:
            return out, sr, qr_seg
        else:
            return out