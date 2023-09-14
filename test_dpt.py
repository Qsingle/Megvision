#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   test_dpt.py
    @Time    :   2023/08/17 10:13:18
    @Author  :   12718 
    @Version :   1.0
'''

import megengine as mge

from megvision.model.segmentation.dpt import DPT

from megengine import random
model = DPT(real_img_size=1024, arch="sam_vit_b", img_size=1024, checkpoint=None)
model.eval()
x = random.normal(size=(1, 3, 1024, 1024))
out = model(x)
print(out.shape)