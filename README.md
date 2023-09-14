# Megvision
The MegEngine vision of the models we used in our experiments, hope this repository can help with you.

# Usage

Sample to import the DPT based on the SAM.
```python
import megengine as mge
from megvision.model.segmentation.dpt import DPT
from megengine import random

model = DPT(real_img_size=1024, arch="sam_vit_b", img_size=1024, checkpoint=None)
model.eval()
x = random.normal(size=(1, 3, 1024, 1024))
out = model(x)
print(out.shape)
```

You can also 

## Supported Model List

+ ### Classification
  + AlexNet: [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997)
  + VGGNet: [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
  + ResNets: [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    + WideResNets: [Wide Residual Networks](https://arxiv.org/pdf/1605.07146.pdf)
    + ResNeXt: [Aggregated Residual Transformation for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
    + SENets: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
    + ResNeSt: [ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf)

+ ### Segmentation:

  + SuperVessel [SuperVessel: Segmenting High-resolution Vessel from Low-resolution Retinal Image](https://arxiv.org/abs/2207.13882)
  + PSPNet [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)
  + SCSNet:[SCS-Net: A Scale and Context Sensitive Network for Retinal Vessel Segmentation](https://www.sciencedirect.com/science/article/pii/S1361841521000712#!)
  + ESPNets:
    + V2:[ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network](https://arxiv.org/pdf/1811.11431.pdf)
  + DeepLab:
    + V3:[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
    + V3Plus:[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)
  + CE-Net:[CE-Net: Context Encoder Network for 2D Medical Image Segmentation](https://arxiv.org/pdf/1903.02740.pdf)
  + CS-Net: [CS2-Net: Deep learning segmentation of curvilinear structures in medical imaging](https://www.sciencedirect.com/science/article/pii/S1361841520302383)
  + PFSeg: [Patch-free 3D Medical Image Segmentation Driven by Super-Resolution Technique and Self-Supervised Guidance](https://link.springer.com/chapter/10.1007/978-3-030-87193-2_13)
  + DPT: [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)
  + SAM: [Segment Anything](https://arxiv.org/abs/2304.02643)
    + Thanks for the MegEngine group to provide the [MegEngine-SAM](https://github.com/MegEngine/MegEngine-SAM) implementation.