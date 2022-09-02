# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:cs2net
    author: 12718
    time: 2021/12/2 15:51
    tool: PyCharm
"""

import megengine.module as M
import megengine.functional as F

from megvision.model.classification.resnet import BasicBlock
from megvision.model.segmentation.unet import DoubleConv
from megvision.layers import ChannelAttentionModule, Conv2d
# try:
#     from megengine.module import PixelShuffle
# except :
#     '''
#         To compatible with megengine < 1.7.0
#     '''
from megvision.layers import PixelShuffle
from megvision.comm.activation import Tanh
from megvision.layers import FIM

from .build_model import SEGMENTATION_REGISTER

class SpatialAttentionModule(M.Module):
    def __init__(self, in_ch):
        super(SpatialAttentionModule, self).__init__()
        self.query_proj = Conv2d(in_ch, in_ch//3, ksize=(1, 3), padding=(0, 1))
        self.key_proj = Conv2d(in_ch, in_ch//3, ksize=(3, 1), padding=(1, 0))
        self.value_proj = M.Conv2d(in_ch, in_ch, kernel_size=1)
        self.gamma = F.zeros(1)

    def forward(self, x):
        bs, c, h, w = x.shape
        key = self.key_proj(x).reshape(bs, -1, h * w)
        query = self.query_proj(x).reshape(bs, -1, h * w)
        value = self.value_proj(x).reshape(bs, c, -1)
        query = query.transpose(0, 2, 1)
        energy = F.matmul(query, key)
        attention = F.softmax(energy)
        out = F.matmul(value, attention.transpose(0, 2, 1)).reshape(bs, c, h, w)
        out = out * self.gamma + x
        return out

@SEGMENTATION_REGISTER.register()
class CSNet(M.Module):
    def __init__(self, in_ch=3, num_classes=2, super_reso=False, output_size=None, upscale_rate=2):
        """
        "CS2-Net: Deep learning segmentation of curvilinear structures in medical imaging"
        <https://www.sciencedirect.com/science/article/pii/S1361841520302383>

        Args:
            in_ch:
            num_classes:
            super_reso:
            output_size:
            upscale_rate:
        """
        super(CSNet, self).__init__()
        self.enc_input = BasicBlock(in_ch, 32, downsample=M.Conv2d(in_ch, 32, 1, 1))
        self.encoder1 = BasicBlock(32, 64, downsample=M.Conv2d(32, 64, 1, 1))
        self.encoder2 = BasicBlock(64, 128, downsample=M.Conv2d(64, 128, 1, 1))
        self.encoder3 = BasicBlock(128, 256, downsample=M.Conv2d(128, 256, 1, 1))
        self.encoder4 = BasicBlock(256, 512, downsample=M.Conv2d(256, 512, 1, 1))

        self.down = M.MaxPool2d(2, 2)

        self.sam = SpatialAttentionModule(512)
        self.cam = ChannelAttentionModule()


        self.decoder4 = DoubleConv(512, 256)
        self.decoder3 = DoubleConv(256, 128)
        self.decoder2 = DoubleConv(128, 64)
        self.decoder1 = DoubleConv(64, 32)

        self.deconv4 = M.ConvTranspose2d(512, 256, 2, 2)
        self.deconv3 = M.ConvTranspose2d(256, 128, 2, 2)
        self.deconv2 = M.ConvTranspose2d(128, 64, 2, 2)
        self.deconv1 = M.ConvTranspose2d(64, 32, 2, 2)

        self.out_conv = M.Conv2d(32, num_classes, 1, 1)
        self.super_reso = super_reso
        self.output_size = output_size
        if super_reso:
            self.upscale_rate = upscale_rate
            self.sr_decoder4 = DoubleConv(512, 256)
            self.sr_decoder3 = DoubleConv(256, 128)
            self.sr_decoder2 = DoubleConv(128, 64)
            self.sr_decoder1 = DoubleConv(64, 32)

            self.sr_deconv4 = M.ConvTranspose2d(512, 256, 2, 2)
            self.sr_deconv3 = M.ConvTranspose2d(256, 128, 2, 2)
            self.sr_deconv2 = M.ConvTranspose2d(128, 64, 2, 2)
            self.sr_deconv1 = M.ConvTranspose2d(64, 32, 2, 2)

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
        down1_0 = self.enc_input(x)
        down1 = self.down(down1_0)
        down2_0 = self.encoder1(down1)
        down2 = self.down(down2_0)
        down3_0 = self.encoder2(down2)
        down3 = self.down(down3_0)
        down4_0 = self.encoder3(down3)
        down4 = self.down(down4_0)
        down5 = self.encoder4(down4)

        sa = self.sam(down5)
        ca = self.cam(down5)
        att_aff = ca + sa

        up4 = self.deconv4(att_aff)
        up4 = F.vision.interpolate(up4, size=down4_0.shape[2:], mode="bilinear", align_corners=True)
        up4 = F.concat([down4_0, up4], axis=1)
        up4 = self.decoder4(up4)

        up3 = self.deconv3(up4)
        up3 = F.vision.interpolate(up3, size=down3_0.shape[2:], mode="bilinear", align_corners=True)
        up3 = F.concat([down3_0, up3], axis=1)
        up3 = self.decoder3(up3)

        up2 = self.deconv2(up3)
        up2 = F.vision.interpolate(up2, size=down2_0.shape[2:], mode="bilinear", align_corners=True)
        up2 = F.concat([down2_0, up2], axis=1)
        up2 = self.decoder2(up2)

        up1 = self.deconv1(up2)
        up1 = F.vision.interpolate(up1, size=down1_0.shape[2:], mode="bilinear", align_corners=True)
        up1 = F.concat([down1_0, up1], axis=1)
        up1 = self.decoder1(up1)

        out = self.out_conv(up1)

        sr = None
        if self.super_reso:
            out = F.vision.interpolate(out, scale_factor=self.upscale_rate, mode="bilinear", align_corners=True)
            if self.training:
                sr_up4 = self.sr_deconv4(att_aff)
                sr_up4 = F.vision.interpolate(sr_up4, size=down4_0.shape[2:], mode="bilinear", align_corners=True)
                sr_up4 = F.concat([down4_0, sr_up4], axis=1)
                sr_up4 = self.sr_decoder4(sr_up4)

                sr_up3 = self.sr_deconv3(sr_up4)
                sr_up3 = F.vision.interpolate(sr_up3, size=down3_0.shape[2:], mode="bilinear", align_corners=True)
                sr_up3 = F.concat([down3_0, sr_up3], axis=1)
                sr_up3 = self.sr_decoder3(sr_up3)

                sr_up2 = self.sr_deconv2(sr_up3)
                sr_up2 = F.vision.interpolate(sr_up2, size=down2_0.shape[2:], mode="bilinear", align_corners=True)
                sr_up2 = F.concat([down2_0, sr_up2], axis=1)
                sr_up2 = self.sr_decoder2(sr_up2)

                sr_up1 = self.sr_deconv1(sr_up2)
                sr_up1 = F.vision.interpolate(sr_up1, size=down1_0.shape[2:], mode="bilinear", align_corners=True)
                sr_up1 = F.concat([down1_0, sr_up1], axis=1)
                sr_up1 = self.sr_decoder1(sr_up1)
                sr = self.sr(sr_up1)
                qr_seg, qr_sr = self.query(sr, out)

        if self.output_size is not None:
            out = F.vision.interpolate(out, size=self.output_size, mode="bilinear", align_corners=True)
            if self.training:
                sr = F.vision.interpolate(sr, size=self.output_size, mode="bilinear", align_corners=True)
        if sr is not None:
            return out, qr_sr, qr_seg
        return out

if __name__ == "__main__":
    import megengine as mge
    x = mge.random.normal(size=(1, 3, 384, 384))
    model = CSNet()
    out = model(x)
    print(out.shape)