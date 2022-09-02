# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    Implementation of the split unet
    filename:spunet
    author: 12718
    time: 2021/12/2 10:10
    tool: PyCharm
"""

import megengine as mge
import megengine.module as M
import megengine.functional as F
from megengine.module.init import zeros_

from megvision.layers import Conv2d
from .unet import DoubleConv

from .build_model import SEGMENTATION_REGISTER

class SplitConv(M.Module):
    def __init__(self, in_ch, out_ch, norm_layer:M.Module=M.InstanceNorm, activation=M.PReLU(), dilations=[1, 2, 4]):
        super(SplitConv, self).__init__()
        hidden_ch = int(out_ch*len(dilations))
        self.identify = None
        if in_ch != out_ch:
            self.identify = Conv2d(in_ch, out_ch, 1, 1, norm_layer=None, activation=None)
        self.conv1 = Conv2d(in_ch, hidden_ch,
                            norm_layer=norm_layer, activation=activation)
        self.conv2 = Conv2d(hidden_ch, hidden_ch, 3, 1, 1, groups=4,
                            norm_layer=norm_layer, activation=activation)
        self.conv3 = []
        self.branchs = len(dilations)
        for i in range(len(dilations)):
            self.conv3.append(
                Conv2d(out_ch, out_ch, 3, 1, padding=dilations[i], dilation=dilations[i],
                       norm_layer=None, activation=None)
            )
        self.bn_act = M.Sequential(
            norm_layer(hidden_ch),
            activation if not isinstance(activation, M.PReLU) else M.PReLU(hidden_ch)
        )
        self.conv4 = Conv2d(hidden_ch, out_ch,
                              norm_layer=norm_layer, activation=None)
        self.activation = activation if not isinstance(activation, M.PReLU) else M.PReLU(out_ch)

        zeros_(self.conv4.norm_layer.weight)

    def forward(self, x):
        net = self.conv1(x)
        identify = x if self.identify is None else self.identify(x)
        net = self.conv2(net)
        splits = F.split(net, self.branchs, axis=1)
        split_outs = []
        for (split, layer) in zip(splits, self.conv3):
            split_outs.append(layer(split))
        concat = F.concat(split_outs, axis=1)
        net = self.bn_act(concat)
        net = self.conv4(net) + identify
        net = self.activation(net)
        return net

@SEGMENTATION_REGISTER.register()
class SpUNet(M.Module):
    def __init__(self, in_ch=3, num_classes=2, dilations=[1, 2, 4], scale=1.0,
                 norm_layer=M.BatchNorm2d, activation=M.LeakyReLU(),
                 super_reso=False, output_size=None):
        super(SpUNet, self).__init__()
        self.down = M.MaxPool2d(2, stride=2)
        base_ch = int(16*scale)
        self.down1_conv = DoubleConv(in_ch, base_ch, norm_layer=norm_layer, activation=activation)
        self.down2_conv = DoubleConv(base_ch, base_ch*2, norm_layer=norm_layer, activation=activation)
        self.down3_conv = DoubleConv(base_ch*2, base_ch*4, norm_layer=norm_layer, activation=activation)
        self.down4_conv = SplitConv(base_ch*4, base_ch*8, norm_layer=norm_layer, activation=activation, dilations=dilations)
        self.down5_conv = SplitConv(base_ch*8, base_ch*16, norm_layer=norm_layer, activation=activation, dilations=dilations)

        self.up5_conv = SplitConv(base_ch*16, base_ch*8, norm_layer=norm_layer, activation=activation, dilations=dilations)
        self.up5_resample = M.Conv2d(base_ch*16, base_ch*8, 1, 1, bias=False)
        self.up6_conv = SplitConv(base_ch*8, base_ch*4, norm_layer=norm_layer,activation=activation, dilations=dilations)
        self.up6_resample = M.Conv2d(base_ch*8, base_ch*4, 1, 1, bias=False)
        self.up7_conv = DoubleConv(base_ch*4, base_ch*2, norm_layer=norm_layer, activation=activation)
        self.up7_resample = M.Conv2d(base_ch*4, base_ch*2, 1, 1, bias=False)
        self.up8_conv = DoubleConv(base_ch*2, base_ch, norm_layer=norm_layer, activation=activation)
        self.up8_resample = M.Conv2d(base_ch*2, base_ch, 1, 1, bias=False)
        self.out_conv = M.Conv2d(base_ch, num_classes, 1, 1, bias=False)

    def forward(self, x):
        down1_0 = self.down1_conv(x)
        down1 = self.down(down1_0)
        down2_0 = self.down2_conv(down1)
        down2 = self.down(down2_0)
        down3_0 = self.down3_conv(down2)
        down3 = self.down(down3_0)
        down4_0 = self.down4_conv(down3)
        down4 = self.down(down4_0)
        down5 = self.down5_conv(down4)

        up5 = F.vision.interpolate(down5, size=down4_0.shape[2:], mode="bilinear", align_corners=True)
        up5 = self.up5_resample(up5)
        up5_concat = F.concat([up5, down4_0], axis=1)
        up5 = self.up5_conv(up5_concat)
        up6 = F.vision.interpolate(up5, size=down3_0.shape[2:], mode="bilinear", align_corners=True)
        up6 = self.up6_resample(up6)
        up6_concat = F.concat([up6, down3_0], axis=1)
        up6 = self.up6_conv(up6_concat)
        up7 = F.vision.interpolate(up6, size=down2_0.shape[2:], mode="bilinear", align_corners=True)
        up7 = self.up7_resample(up7)
        up7_concat = F.concat([up7, down2_0], axis=1)
        up7 = self.up7_conv(up7_concat)
        up8 = F.vision.interpolate(up7, size=down1_0.shape[2:], mode="bilinear", align_corners=True)
        up8 = self.up8_resample(up8)
        up8_concat = F.concat([up8, down1_0], axis=1)
        up8 = self.up8_conv(up8_concat)
        out = self.out_conv(up8)
        return out

if __name__ == "__main__":
    import megengine as mge
    x = mge.random.normal(size=(2, 3, 512, 512))
    layer = SpUNet(3, num_classes=2)
    out = layer(x)
    print(out.shape)