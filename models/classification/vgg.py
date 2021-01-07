#-*- coding:utf-8 -*-
#!/etc/env python
'''
   @Author:Zhongxi Qiu
   @File: vgg.py
   @Time: 2020-12-31 10:55:37
   @Version:1.0
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import megengine as mge
import megengine.module as M
import megengine.functional as F


model_urls = {
    'vgg11': '',
    'vgg13': '',
    'vgg16': '',
    'vgg19': '',
    'vgg11_bn': '',
    'vgg13_bn': '',
    'vgg16_bn': '',
    'vgg19_bn': '',
}

class VGG(M.Module):
    def __init__(self, cfg, num_classes=1000, in_channels=3, init_weights=True, batch_norm=False):
        '''
            VGGNet from paper
            "Very Deep Convolutional Networks For Large-Scale Image Recognition"<https://arxiv.org/pdf/1409.1556.pdf>
        '''
        super(VGG, self).__init__()
        self.features = self._make_layers(in_channels, cfg, batch_norm)
        self.avgpool = M.AdaptiveAvgPool2d((7,7))
        self.classifier = M.Sequential(
            M.Linear(512*7*7, 4096),
            M.ReLU(),
            M.Dropout(),
            M.Linear(4096, 4096),
            M.ReLU(),
            M.Dropout(),
            M.Linear(4096, num_classes)
        )

        if init_weights:
            self._init_weights()

    def _make_layers(self, in_channels, cfg, batch_norm=False):
        '''
            Make the layer from the config.
            Args:
                in_channels(int): the number of channels of first conv layer.
                cfg(list):  the config of the layers.
                batch_norm(bool): If true use the batch normalization after the conv2d layer
        '''
        layers = []
        in_ch = in_channels
        for v in cfg:
            if v == "M":
                layers.append(M.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = M.Conv2d(in_ch, v, kernel_size=3, stride=1, padding=1)
                if batch_norm:
                    layers += [conv2d, M.BatchNorm2d(v), M.ReLU()]
                else:
                    layers += [conv2d, M.ReLU()]
                in_ch = v
        return M.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = F.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight)
                if m.bias is not None:
                    M.init.zeros_(m.bias)
            elif isinstance(m, M.BatchNorm2d):
                M.init.fill_(m.weight, 1)
                M.init.zeros_(m.bias)
            elif isinstance(m, M.Linear):
                M.init.normal_(m.weight, 0, 0.01)
                M.init.zeros_(m.bias)

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(pretrained=False, progress=True, cfg_arch="A", arch="vgg11",**kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(cfgs[cfg_arch],**kwargs)

    if pretrained:
        state_dict = mge.hub.load_serialized_obj_from_url(model_urls[arch])
        model.load_state_dict(state_dict)
    return model

def vgg11(pretrained=False, progress=True, **kwargs):
    '''
        VGG 11-layer model (cfg "A").
        Args:
            pretrained (bool): If True, return a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    '''
    kwargs["batch_norm"] = False
    return _vgg(pretrained, progress, "A", "vgg11", **kwargs)

def vgg11_bn(pretrained=False, progress=True, **kwargs):
    '''
        VGG 11-layer model  (cfg "A") with batch normalization.
        Args:
            pretrained (bool): If True, return a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    '''
    kwargs["batch_norm"] = True
    return _vgg(pretrained, progress, "A", "vgg11_bn", **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    '''
        VGG 13-layer model (cfg "B").
        Args:
            pretrained (bool): If True, return a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    '''
    kwargs["batch_norm"] = False
    return _vgg(pretrained, progress, "B", "vgg13", **kwargs)

def vgg13_bn(pretrained=False, progress=True, **kwargs):
    '''
        VGG 13-layer model  (cfg "B") with batch normalization.
        Args:
            pretrained (bool): If True, return a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    '''
    kwargs["batch_norm"] = True
    return _vgg(pretrained, progress, "B", "vgg13_bn", **kwargs)

def vgg16(pretrained=False, progress=True, **kwargs):
    '''
        VGG 16-layer model (cfg "D").
        Args:
            pretrained (bool): If True, return a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    '''
    kwargs["batch_norm"] = False
    return _vgg(pretrained, progress, "D", "vgg16", **kwargs)

def vgg16_bn(pretrained=False, progress=True, **kwargs):
    '''
        VGG 16-layer model (cfg "D") with batch normalization.
        Args:
            pretrained (bool): If True, return a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    '''
    kwargs["batch_norm"] = True
    return _vgg(pretrained, progress, "D", "vgg16_bn", **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    '''
        VGG 16-layer model (cfg "E").
        Args:
            pretrained (bool): If True, return a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    '''
    kwargs["batch_norm"] = False
    return _vgg(pretrained, progress, "E", "vgg19", **kwargs)

def vgg19_bn(pretrained=False, progress=True, **kwargs):
    '''
        VGG 19-layer model (cfg "E") with batch normalization.
        Args:
            pretrained (bool): If True, return a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    '''
    kwargs["batch_norm"] = True
    return _vgg(pretrained, progress, "E", "vgg19_bn", **kwargs)

if __name__ == "__main__":
    model = vgg19_bn()
    x = mge.random.normal(size=(1, 3, 224, 224))
    model.eval()
    print(model)
    pred = model(x)
    print(pred.shape)