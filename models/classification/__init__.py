# -*- coding:utf-8 -*-
# !/usr/bin/env python

from .alexnet import *
from .densenet import *
from .efficientnet import *
from .genet import *
from .googlenet import *
from .inceptionnetv3 import *
from .mobilenetv3 import *
from .nasnet import *
from .resnet import *
from .squeezenet import *
from .vgg import *

model_list_dict = {
    "efficientnet_b0":efficientnet_b0,
    "efficientnet_b1":efficientnet_b1,
    "efficientnet_b2":efficientnet_b2,
    "efficientnet_b3":efficientnet_b3,
    "efficientnet_b4":efficientnet_b4,
    "efficientnet_b5":efficientnet_b5,
    "efficientnet_b6":efficientnet_b6,
    "efficientnet_b7":efficientnet_b7,
    "efficientnet_b8":efficientnet_b7,
    "resnet18":resnet18,
    'resnet34':resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d':resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,
    'seresnet18':seresnet18,
    'seresnet34':seresnet34,
    'seresnet50':seresnet50,
    'seresnet101':seresnet101,
    'seresnet152':seresnet152,
    'seresnext50_32x4d':seresnext50_32x4d,
    'seresnext101_32x8d':seresnext101_32x8d,
    'resnest14': resnest14,
    'resnest26': resnest26,
    'resnest50':resnest50,
    'resnest101':resnest101,
    'resnest200':resnest200,
    'resnest269' :resnest269,
    'genet_small' : genet_small,
    'genet_normal' : genet_normal,
    'genet_large' : genet_large,
    'vgg11':vgg11,
    'vgg11_bn':vgg11_bn,
    'vgg13':vgg13,
    'vgg13_bn':vgg13_bn,
    'vgg16':vgg16,
    'vgg16_bn':vgg16_bn,
    'vgg19':vgg19,
    'vgg19_bn':vgg19_bn,
    'alexnet':alexnet,
    'squeezenet_v1_0':squeezenet_v1_0,
    'squeezenet_v1_1': squeezenet_v1_1,
    'squeezeresnet_v1_0':squeezeresnet_v1_0,
    'squeezeresnet_v1_1':squeezenet_v1_1,
    'nasnetalarge': nasnetalarge,
    'densenet121' : densenet121,
    'densnet161': densenet161,
    'densenet169' : densenet169,
    'densnet201':densenet201,
    'mobilenetv3_small_w7d20' : mobilenetv3_small_w7d20,
    'mobilenetv3_small_wd2' : mobilenetv3_small_wd2,
    'mobilenetv3_small_w3d4' : mobilenetv3_small_w3d4,
    'mobilenetv3_small_w1':mobilenetv3_small_w1,
    'mobilenetv3_small_w5d4' : mobilenetv3_small_w5d4,
    'mobilenetv3_large_w7d20' : mobilenetv3_large_w7d20,
    'mobilenetv3_large_wd2' : mobilenetv3_large_wd2,
    'mobilenetv3_large_w3d4' : mobilenetv3_large_w3d4,
    'mobilenetv3_large_w1' : mobilenetv3_large_w1,
    'mobilenetv3_large_w5d4' : mobilenetv3_large_w5d4,
    'inception_v3':get_inception_v3
}

def create_model(model_name, **kwargs):
    """
    Create model by model name.
    args;
        model_name (str): the name of model
    return:
        A model, M.Module
    """
    return model_list_dict[model_name](**kwargs)