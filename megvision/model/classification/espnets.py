# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:espnets
    author: 12718
    time: 2021/11/18 14:17
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import math

import megengine.module as M
from megengine.module import init
import megengine.functional as F

from layers import Conv2d
from .build import BACKBONE_REGISTER

__all__ = ["EESP", "SESSP", "EspNetV2"]

model_urls = {
    "espnetv2_s_0_5" : "",
    "espnetv2_s_1_0" : "",
    "espnetv2_s_1_25" : "",
    "espnetv2_s_1_5" : "",
    "epsnetv2_s_2_0" : ""
}

class EESP(M.Module):
    def __init__(self, in_ch, out_ch, stride=1, r_lim=7, K=4):
        """
        Implementation of the Extremely Efficient Spatial Pyramid module introduced in
        "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
        <https://arxiv.org/pdf/1811.11431.pdf>
        Parameters
        ----------
        in_ch (int): number of channels for input
        out_ch (int): number of channels for output
        stride (int): stride of the convs
        r_lim (int): A maximum value of receptive field allowed for EESP block
        K (int): number of parallel branches
        """
        super(EESP, self).__init__()
        hidden_ch = int(out_ch // K)
        hidden_ch1 = out_ch - hidden_ch*(K-1)
        assert hidden_ch1 == hidden_ch, \
            "hidden size of n={} must equal to hidden size of n1={}".format(hidden_ch, hidden_ch1)
        self.g_conv1 = Conv2d(in_ch, hidden_ch, 1, stride=1,
                              groups=K, activation=M.PReLU(hidden_ch))

        self.spp_convs = []
        for i in range(K):
            ksize = int(3 + i * 2)
            dilation = int((ksize - 1) / 2) if ksize <= r_lim else 1
            self.spp_convs.append(M.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=dilation, dilation=dilation, groups=hidden_ch, bias=False))

        self.conv_concat = Conv2d(out_ch, out_ch, groups=K, activation=None)
        self.bn_pr = M.Sequential(
            M.BatchNorm2d(out_ch),
            M.PReLU(out_ch)
        )
        self.module_act = M.PReLU(out_ch)
        self.K = K
        self.stride = stride

    def forward(self, x):
        net = self.g_conv1(x)
        outputs = [self.spp_convs[0](net)]
        for i in range(1, self.K):
            output_k = self.spp_convs[i](net)
            output_k = output_k + outputs[i-1]
            outputs.append(
                output_k
            )
        concat = F.concat(outputs, axis=1)
        concat = self.bn_pr(concat)
        net = self.conv_concat(concat)
        if self.stride == 2:
            return net
        if net.size == x.size:
            net = net + x
        net = self.module_act(net)
        return net

class SESSP(M.Module):
    def __init__(self, in_ch, out_ch, stride=2, r_lim=7, K=4, refin=True, refin_ch=3):
        """
            Implementation of the Extremely Efficient Spatial Pyramid module introduced in
            "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
            <https://arxiv.org/pdf/1811.11431.pdf>
            Parameters
            ----------
            in_ch (int): number of channels for input
            out_ch (int): number of channels for output
            stride (int): stride of the convs
            r_lim (int): A maximum value of receptive field allowed for EESP block
            K (int): number of parallel branches
            refin (bool): whether use the inference from input image
        """
        super(SESSP, self).__init__()
        eesp_out = out_ch - in_ch
        self.eesp = EESP(in_ch, eesp_out, stride=stride, r_lim=r_lim, K=K)
        self.avg_pool = M.AvgPool2d(3, stride=stride, padding=1)
        self.refin = refin
        self.stride = stride
        self.activation = M.PReLU(out_ch)
        if refin:
            self.refin_conv = M.Sequential(
                Conv2d(refin_ch, refin_ch, ksize=3, stride=1, padding=1, activation=M.PReLU(refin_ch)),
                Conv2d(refin_ch, out_ch, activation=None)
            )


    def forward(self, inputs, _ref=None):
        avgout = self.avg_pool(inputs)
        eesp_out = self.eesp(inputs)
        net = F.concat([eesp_out, avgout], axis=1)
        if self.refin:
            w1 = avgout.shape[2]
            w2 = _ref.shape[2]
            while w2 != w1:
                _ref = F.avg_pool2d(_ref, kernel_size=3, stride=self.stride, padding=1)
                w2 = _ref.shape[2]
            _ref = self.refin_conv(_ref)
            net = net + _ref
        net = self.activation(net)
        return net

class EspNetV2(M.Module):
    def __init__(self, in_ch=3, num_classes=1000, scale=1.0):
        """
            Implementation of the ESPNetV2 introduced in
            "ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network"
            <https://arxiv.org/pdf/1811.11431.pdf>
            Parameters
            ----------
            in_ch (int): number of channels for input
            num_classes (int): number of classes
            scale (float): the scale rate for the net
        """
        super(EspNetV2, self).__init__()
        reps = [0, 3, 7, 3] #how many times the essp block repeat
        r_lims = [13, 11, 9, 7, 5]
        K = [4] * len(r_lims)

        base = 32
        config_len = 5
        config = [base] * config_len
        base_s = 0
        for i in range(config_len):
            if i == 0:
                base_s = int(base * scale)
                base_s = math.ceil(base_s/ K[0]) * K[0]
                config[i] = base if base_s > base else base_s
            else:
                config[i] = base_s * pow(2, i)
        if scale <= 1.5:
            config.append(1024)
        elif scale <= 2.0:
            config.append(1280)
        else:
            ValueError("Configuration for scale={} not supported".format(scale))

        ref_input = in_ch
        self.reinf = True

        self.level1 = Conv2d(in_ch, config[0], 3, stride=2, padding=1, activation=M.PReLU(config[0]))
        self.level2_0 = SESSP(config[0], config[1], stride=2, r_lim=r_lims[0], K=K[0],
                            refin=self.reinf, refin_ch=ref_input)

        self.level3_0 = SESSP(config[1], config[2], stride=2, r_lim=r_lims[1], K=K[1],
                              refin=self.reinf, refin_ch=ref_input)

        self.level3 = []
        for i in range(reps[1]):
            self.level3.append(EESP(config[2], config[2], stride=1, r_lim=r_lims[2], K=K[2]))

        self.level4_0 = SESSP(config[2], config[3], stride=2, r_lim=r_lims[2], K=K[2],
                              refin=self.reinf, refin_ch=ref_input)
        self.level4 = []
        for i in range(reps[2]):
            self.level4.append(EESP(config[3], config[3], stride=1, r_lim=r_lims[3], K=K[3]))

        self.level5_0 = SESSP(config[3], config[4], stride=2, r_lim=r_lims[3], K=K[3],
                              refin=self.reinf, refin_ch=ref_input)
        self.level5 = []
        for i in range(reps[3]):
            self.level5.append(EESP(config[4], config[4], stride=1, r_lim=r_lims[4], K=K[4]))

        self.level5.append(Conv2d(config[4], config[4], ksize=3, stride=1, padding=1,
                                  groups=config[4], activation=M.PReLU(config[4])))
        self.level5.append(Conv2d(config[4], config[5], ksize=1, stride=1, padding=0,
                                  groups=K[3], activation=M.PReLU(config[5])))
        self.classifier = M.Linear(config[5], num_classes)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, M.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, M.BatchNorm2d):
                init.fill_(m.weight, 1)
                init.zeros_(m.bias)
            elif isinstance(m, M.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, inputs):
        out_l1 = self.level1(inputs)
        if not self.reinf:
            del inputs
            inputs = None
        out_l2 = self.level2(out_l1, inputs)

        out_l3_0 = self.level3_0(out_l2, inputs)
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        outl4_0 = self.level4_0(out_l3, inputs)
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(outl4_0)
            else:
                out_l4 = layer(out_l4)

        outl5_0 = self.level5_0(out_l4, inputs)
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(outl5_0)
            else:
                out_l5 = layer(out_l5)
        net = F.adaptive_avg_pool2d(out_l5, 1)
        net = F.flatten(net, 1)
        net = self.classifier(net)
        return net

def _espnerv2(arch, pretrained=False, **kwargs):
    model = EspNetV2(**kwargs)
    if pretrained:
        if model_urls[arch] == '':
            print("The weights file of {} is not provided now, pass to load pretrained weights".format(arch))
        else:
            state_dict = mge.hub.load_serialized_obj_from_url(model_urls[arch], model_dir="./weights")
            model.load_state_dict(state_dict)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_0_5(pretrained=False, **kwargs):
    kwargs["scale"] = 0.5
    model = _espnerv2("espnetv2_s_0_5", pretrained=pretrained, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_1_0(pretrained=False, **kwargs):
    kwargs["scale"] = 1.0
    model = _espnerv2("espnetv2_s_1_0", pretrained=pretrained, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_1_25(pretrained=False, **kwargs):
    kwargs["scale"] = 1.25
    model = _espnerv2("espnetv2_s_1_25", pretrained=pretrained, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_1_5(pretrained=False, **kwargs):
    kwargs["scale"] = 1.5
    model = _espnerv2("espnetv2_s_1_5", pretrained=pretrained, **kwargs)
    return model

@BACKBONE_REGISTER.register()
def espnetv2_s_2_0(pretrained=False, **kwargs):
    kwargs["scale"] = 2.0
    model = _espnerv2("espnetv2_s_2_0", pretrained=pretrained, **kwargs)
    return model


if __name__ == "__main__":
    import megengine as mge
    x = mge.random.normal(size=(1, 3, 224, 224))

    model = EspNetV2(3, 1000, 0.5)
    out = model(x)
    print(out.shape)