#-*- coding:utf-8 -*-
#!/etc/env python
'''
   @Author:Zhongxi Qiu
   @File: utils.py
   @Time: 2021-01-01 15:34:15
   @Version:1.0
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import megengine.module.init as init
import megengine.module as M
import math

class SEModule(M.Module):
    '''
        Implementation of semodule in SENet and MobileNetV3, there we use 1x1 conv replace the linear layer.
        SENet:"Squeeze-and-Excitation Networks"<https://arxiv.org/abs/1709.01507>
        MobileNetV3: "Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>
    '''
    def __init__(self, in_ch, reduction=16,  norm_layer=None, nolinear=M.ReLU(), sigmoid=M.Sigmoid()):
        '''
            Initialize the module.
            @in_ch: int, the number of channels of input,
            @reduction: int, the coefficient of dimensionality reduction
            @sigmoid: M.Module, the sigmoid function, in MobilenetV3 is H-Sigmoid and in SeNet is sigmoid
            @norm_layer: M.Module, the batch normalization moldule
            @nolinear: M.Module, the nolinear function module
            @sigmoid: M.Module, the sigmoid layer
        '''
        super(SEModule, self).__init__()
        if norm_layer is None:
            norm_layer = M.BatchNorm2d

        if nolinear is None:
            nolinear = M.ReLU()
        
        if sigmoid is None:
            sigmoid = M.Sigmoid()

        self.avgpool = M.AdaptiveAvgPool2d(1)
        self.fc = M.Sequential(
            M.Conv2d(in_ch, in_ch // reduction, kernel_size=1, stride=1, padding=0),
            norm_layer(in_ch // reduction),
            nolinear,
            M.Conv2d(in_ch // reduction, in_ch, kernel_size=1, stride=1, padding=0),
            norm_layer(in_ch),
            sigmoid,
        )

    def forward(self, x):
        net = self.avgpool(x)
        net = self.fc(net)
        return net

def kaiming_norm_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''
        Kaiming initilization.
        Args:
            tensor: an n-dimensional megengine.Tensor
            a: the negtive slope
            mode: either 'fan_in' or 'fan_out'
            nonlinearity: the non-linear function (the name of the non-linear function)
    '''
    fan = init.calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return init.normal_(tensor, 0, std)

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    '''
        Kaiming initilization.
        Args:
            tensor: an n-dimensional megengine.Tensor
            a: the negtive slope
            mode: either 'fan_in' or 'fan_out'
            nonlinearity: the non-linear function (the name of the non-linear function)
    '''
    fan = init.calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) / math.sqrt(fan)
    return init.uniform_(tensor, -bound, bound) 