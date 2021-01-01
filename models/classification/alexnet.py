#-*- coding:utf-8 -*-
#!/etc/env python
'''
   @Author:Zhongxi Qiu
   @File: alexnet.py
   @Time: 2020-12-31 09:35:29
   @Version:1.0
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import megengine as mge
import megengine.module as M

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    "alexnet":""
}

class AlexNet(M.Module):
    def __init__(self, in_ch=3, num_classes=1000):
        '''
            The AlexNet.
            args:
                in_ch: int, the number of channels of inputs
                num_classes: int, the number of classes that need to predict
            reference:
                "One weird trick for parallelizing convolutional neural networks"<https://arxiv.org/abs/1404.5997>
        '''
        super(AlexNet, self).__init__()
        #the part to extract feature
        self.features = M.Sequential(
            M.Conv2d(in_ch, 64, kernel_size=11, stride=4, padding=11//4),
            M.ReLU(),
            M.MaxPool2d(kernel_size=3, stride=2),
            M.Conv2d(64, 192, kernel_size=5, padding=2),
            M.ReLU(),
            M.MaxPool2d(kernel_size=3, stride=2),
            M.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            M.ReLU(),
            M.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            M.ReLU(),
            M.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            M.ReLU(),
            M.MaxPool2d(kernel_size=3, stride=2),
        )
        #global avg pooling
        self.avgpool = M.AdaptiveAvgPool2d((6,6))
        #classify part
        self.classifier = M.Sequential(
            M.Dropout(),
            M.Linear(256*6*6, 4096),
            M.ReLU(),
            M.Dropout(),
            M.Linear(4096, 4096),
            M.ReLU(),
            M.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = mge.functional.flatten(x, 1)
        x = self.classifier(x)
        return x

def alexnet(pretrained=False, progress=True, **kwargs):
    """
        AlexNet model.
        Args:
            pretrained (bool):If True, returns a model pre-trained on ImageNet
            progress (bool): If True, display a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        pass

    return model

if __name__ == "__main__":
    net = alexnet(in_ch=3)
    x = mge.random.normal(size=[1, 3, 224, 224])
    net.eval()
    pred = net(x)
    print(pred.shape)