#-*- coding:utf-8 -*-
#!/usr/bin/env python
'''
    @File    :   dpt.py
    @Time    :   2023/08/17 09:55:41
    @Author  :   12718 
    @Version :   1.0
'''

import megengine as mge
import megengine.module as M
import megengine.functional.vision as fv
import megengine.functional as F
from typing import List, Union

from megvision.model.classification import BACKBONE_REGISTER
from megvision.model.classification import get_model
from megvision.model.segmentation.mge_segment_anything import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

def update_cfg(default:dict, cfg:dict):
    """AI is creating summary for update_cfg

    Args:
        default (dict): [description]
        cfg (dict): [description]
    Return:
        dict: The updated config
    """
    new_cfg = {}
    for k in cfg.keys():
        if k not in default:
            continue
        else:
            new_cfg[k] = cfg[k]
    default.update(new_cfg)
    return default


DEFAULT_CFG = {
    "dino_vit_tiny": {
        "patch_size": 16,
        "img_size": [224],
    },
    "dino_vit_base": {
        "patch_size": 16,
        "img_size": [224],
    },
    "dino_vit_small": {
        "patch_size": 16,
        "img_size": [224],
    },
    "dinov2_vit_small":  {
              "patch_size": 14,
              "drop_path_rate": 0.4,
              "ffn_layer": "mlp",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
        
            },
    "dinov2_vit_large":  {
              "patch_size": 14,
              "drop_path_rate": 0.4,
              "ffn_layer": "mlp",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
        
            },
    "dinov2_vit_base": {
              "patch_size": 14,
              "drop_path_rate": 0.4,
              "ffn_layer": "mlp",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
    },
    "dinov2_vit_giant2": {
              "patch_size": 14,
              "drop_path_rate": 0.4,
              "ffn_layer": "swiglufused",
              "block_chunks": 0,
              "img_size": 518,
              "init_values": 1e-5
    },
    "beit_vit_base": {
        "patch_size": 16,
        "img_size": 224,
        "num_heads": 12,
    },
    "beit_vit_large": {
            "patch_size": 16,
            "img_size": 224,
            "num_heads": 16,
        },
    "sam_vit_b": {
        "patch_size": 16,
        "img_size": 1024,
        "embed_dim": 768
    },
    "sam_vit_l": {
        "patch_size": 16,
        "img_size": 1024,
        "embed_dim": 1024
    },
    "sam_vit_h": {
        "patch_size": 16,
        "img_size": 1024,
        "embed_dim": 1280
    }
}

class Resample(M.Module):
    def __init__(self, scale, dim, embedding_dim=256, patch_size=16) -> None:
        super(Resample, self).__init__()

        stride = scale // patch_size
        stride = scale // 16
        self.proj = M.Conv2d(dim, embedding_dim, 1, 1)
        if stride < 1:
            stride = 16 // scale
            self.conv = M.ConvTranspose2d(
                embedding_dim, embedding_dim,
                stride, stride
            )
        else:
            self.conv = M.Conv2d(
                embedding_dim, embedding_dim,
                3, stride, 1, bias=False
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv(x)
        return x


class Readout(M.Module):
    def __init__(self, type="proj", dim=384, sam=False) -> None:
        super().__init__()
        self.type = type
        if type == "proj":
            self.proj = M.Sequential(
                M.Linear(2 * dim, dim),
                M.GELU()
            )
        self.sam = sam
        if self.sam:
            self.type = "ignore"

    def forward(self, x):
        if self.type == "ignore":
            if self.sam:
                return x
            return x[0]
        elif self.type == "add":
            return x[0] + x[1]
        elif self.type == "proj":
            img_token = x[0]
            cls_token = x[1]
            cls_token = cls_token.unsqueeze(1).expand_as(img_token)
            feature = F.concat([img_token, cls_token], axis=2)
            feature = self.proj(feature)
            return feature
        else:
            raise ValueError("Unsupported readout way {}".format(self.type))


class ResidualConv(M.Module):
    def __init__(self, ch, use_bn=True) -> None:
        super(ResidualConv, self).__init__()
        self.conv1 = M.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.conv2 = M.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1 = M.Identity()
        self.bn2 = M.Identity()
        if use_bn:
            self.bn1 = M.BatchNorm2d(ch)
            self.bn2 = M.BatchNorm2d(ch)
        self.activation = M.ReLU()

    def forward(self, x):
        residual = x
        net = self.activation(x)
        net = self.conv1(net)
        net = self.bn1(net)
        net = self.activation(net)
        net = self.conv2(net)
        net = self.bn2(net)
        net = net + residual
        return net


class RefineFusion(M.Module):
    def __init__(self, ch, use_bn=True) -> None:
        super(RefineFusion, self).__init__()
        self.residual1 = ResidualConv(ch, use_bn=use_bn)
        self.residual2 = ResidualConv(ch, use_bn=use_bn)
        self.proj = M.Conv2d(ch, ch, 1, 1, 0, bias=True)

    def forward(self, *xs):
        assert len(xs) == 2, "The number of input features must be 2, but got {}".format(len(xs))
        x1, x2 = xs
        net = self.residual1(x1) + x2
        net = self.residual2(net)
        net = fv.interpolate(net, scale_factor=2, mode="bilinear", align_corners=True)
        net = self.proj(net)
        return net


class Interpolate(M.Module):
    def __init__(self, scale_factor=None, size=None, mode="nearest", align_corners=None) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return fv.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners)

class DPTDecoder(M.Module):
    def __init__(self, embed_dim, image_size,
                 fea_dims=[96, 192, 384, 768], fusion_dim=256, readout_way="ignore",
                 use_bn=True, patch_size=16, sam=False):
        super(DPTDecoder, self).__init__()
        self.read_outs = [Readout(readout_way, dim=embed_dim, sam=sam) for _ in range(len(fea_dims))]
        self.sam = sam
        self.p_size = image_size // patch_size
        self.scales = [4, 8, 16, 32]
        self.resamples =[
            Resample(s, embed_dim, embedding_dim=embd_dim, patch_size=patch_size) for s, embd_dim in
             zip(self.scales, fea_dims)
        ]
        self.post_proj = [M.Conv2d(d, fusion_dim, 1, 1, bias=True) for d in fea_dims]

        self.refine_1 = ResidualConv(fusion_dim, use_bn=use_bn)
        self.refine_2 = RefineFusion(fusion_dim, use_bn=use_bn)
        self.refine_3 = RefineFusion(fusion_dim, use_bn=use_bn)
        self.refine_4 = RefineFusion(fusion_dim, use_bn=use_bn)

    def forward(self, xs):
        layer1, layer2, layer3, layer4 = xs
        if self.sam:
            bs, dim, p_size, p_size = layer1.shape
        else:
            bs, seq_len, dim = layer1[0].shape
        if self.sam:
            layer4 = self.read_outs[3](layer4)
        else:
            layer4 = self.read_outs[3](layer4).transpose(1, 2).reshape(bs, dim, self.p_size, self.p_size)
        layer4 = self.resamples[3](layer4)
        layer4 = self.post_proj[3](layer4)
        net = self.refine_1(layer4)
        net = fv.interpolate(net, size=(self.p_size, self.p_size), mode="bilinear", align_corners=True)
        if self.sam:
            layer3 = self.read_outs[2](layer3)
        else:
            layer3 = self.read_outs[2](layer3).transpose(1, 2).reshape(bs, dim, self.p_size, self.p_size)
        layer3 = self.resamples[2](layer3)
        layer3 = self.post_proj[2](layer3)
        net = self.refine_2(layer3, net)
        if self.sam:
            layer2 = self.read_outs[1](layer2)
        else:
            layer2 = self.read_outs[1](layer2).transpose(1, 2).reshape(bs, dim, self.p_size, self.p_size)
        layer2 = self.resamples[1](layer2)
        layer2 = self.post_proj[1](layer2)
        net = self.refine_3(layer2, net)
        if self.sam:
            layer1 = self.read_outs[0](layer1)
        else:
            layer1 = self.read_outs[0](layer1).transpose(1, 2).reshape(bs, dim, self.p_size, self.p_size)
        layer1 = self.resamples[0](layer1)
        layer1 = self.post_proj[0](layer1)
        net = self.refine_4(layer1, net)
        return net

sam_build_func = {
    "sam_vit_b": build_sam_vit_b,
    "sam_vit_l":  build_sam_vit_l,
    "sam_vit_h": build_sam_vit_h
}

class DPT(M.Module):
    def __init__(self, arch:str="dino_vit_base", inter_blocks:List[int]=[3, 6, 9, 12], in_ch=3, out_dim=1,
                 fea_dims=[96, 192, 384, 768], fusion_dim=256, drop_rate=0.1, non_negtive=True,
                 real_img_size=None, readout_way="ignore", use_bn=True, task="depth",
                 checkpoint=None, key=None, upscale_rate=2, super_reso=False, aux=False, **kwargs):
        super(DPT, self).__init__()
        supported_arch = [name for name in BACKBONE_REGISTER.get_names() if "vit" in name]
        supported_arch += ["sam_vit_b", "sam_vit_l", "sam_vit_h"]
        print(supported_arch)
        default_cfg = DEFAULT_CFG[arch]
        cfg = update_cfg(default_cfg, kwargs)
        self.sam = False
        assert arch in supported_arch, "We only support visual transformer architechture now, excepht in {}, but got {}".format(supported_arch, arch)
        if arch.startswith("sam"):
            self.encoder = sam_build_func[arch](checkpoint=None).image_encoder
            self.reset_backbone(checkpoint, key=key)
            self.sam = True
        else:
            self.encoder = get_model(arch)(**cfg)
            self.reset_backbone(checkpoint, key)
        image_size = real_img_size or cfg['img_size']
        if isinstance(image_size, list):
            image_size = image_size[0]
        if arch.startswith("sam"):
            dim = cfg["embed_dim"]
            self.patch_size = cfg["patch_size"]
        else:
            dim = self.encoder.embed_dim
            self.patch_size = self.encoder.patch_embed.patch_size
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]
        self.get_blocks = [i - 1 for i in inter_blocks]
        self.decoder = DPTDecoder(dim, image_size, fea_dims=fea_dims, fusion_dim=fusion_dim,
                                 readout_way=readout_way, use_bn=use_bn, patch_size=self.patch_size, sam=self.sam)
        self.aux = aux
        if self.aux:
            self.aux_head = M.Conv2d(768, out_dim, 1, 1, 0)
        if super_reso:
            self.sr_decoder = DPTDecoder(
                dim, image_size, fea_dims=fea_dims, fusion_dim=fusion_dim,
                readout_way=readout_way, use_bn=use_bn, patch_size=self.patch_size
            )
            self.sr_head = M.Sequential(
                M.Conv2d(fusion_dim, fusion_dim // 2, 3, 1, 1, bias=False),
                Interpolate(size=(real_img_size, real_img_size), mode="bilinear", align_corners=True),
                M.Conv2d(fusion_dim // 2, 64, 5, 1, 2, bias=False),
                M.ReLU(),
                M.Conv2d(64, 32, 3, 1, 1, bias=False),
                M.ReLU(),
                M.Conv2d(32, (upscale_rate ** 2) * in_ch, kernel_size=3, stride=1, padding=1, bias=False),
                M.PixelShuffle(upscale_factor=upscale_rate)
            )
        if task == "depth":
            self.head = M.Sequential(
                M.Conv2d(fusion_dim, fusion_dim // 2, 3, 1, 1, bias=False),
                Interpolate(size=(real_img_size, real_img_size), mode="bilinear", align_corners=True),
                M.Conv2d(fusion_dim // 2, 32, 3, 1, 1, bias=False),
                M.ReLU(),
                M.Conv2d(32, out_dim, 1, 1),
                M.ReLU() if non_negtive else M.Identity()
            )
        elif task == "seg":
            self.head = M.Sequential(
                M.Conv2d(fusion_dim, fusion_dim, 3, 1, 1, bias=False),
                M.BatchNorm2d(fusion_dim),
                M.ReLU(True),
                M.Dropout(drop_rate),
                M.Conv2d(fusion_dim, out_dim, 1, 1),
                Interpolate(size=(real_img_size, real_img_size), mode="bilinear", align_corners=True)
            )
        else:
            raise ValueError("Unsupport task mode {}".format(task))


    def reset_backbone(self, chekpoint=None, key=None):
        if chekpoint is None:
            return
        if hasattr(self.encoder, "load_checkpoint"):
            self.encoder.load_checkpoint(chekpoint)
        else:
            state = mge.load(chekpoint, map_location="cpu")
            if key is not None:
                state = state[key]
            self.encoder.load_state_dict(state)

    def encode(self, x):
        if self.sam:
            features = self.encoder.get_intermediate_layers(x, self.get_blocks, transpose=True)
        else:
            features = self.encoder.get_intermediate_layers(x, self.get_blocks, reshape=False, return_class_token=True,
                                                        norm=False)
        return features
    def forward(self, x):
        layer1, layer2, layer3, layer4 = self.encode(x)
        net = self.decoder([layer1, layer2, layer3, layer4])
        net = self.head(net)
        if self.aux and self.training:
            bs, ch, h, w = x.shape
            bs, _, dim = layer1[0].shape
            p_size = h // self.patch_size
            layer2 = layer2[0].transpose(1, 2).reshape(bs, dim, p_size, p_size)
            aux_out = fv.interpolate(layer2, size=(h, w), mode="bilinear", align_corners=True)
            aux_out = self.aux_head(aux_out)
            return net, aux_out
        return net


if __name__ == "__main__":
    from megengine import random
    model = DPT(real_img_size=1024, arch="sam_vit_b", img_size=1024, checkpoint=None)
    x = random.normal(size=(1, 3, 1024, 1024))
    out = model(x)
    print(out.shape)