# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :  cityscapes_dataset.py
@Time    :  2021/8/30 14:26
@Author  :  Zhongxi Qiu
@Contact :  qiuzhongxi@163.com
@License :  (C)Copyright 2019-2021
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from megengine.data.dataset import Dataset
from albumentations import Compose
from albumentations import ColorJitter, ShiftScaleRotate
from albumentations import ChannelShuffle, ChannelDropout
from albumentations import Resize, Normalize,PadIfNeeded
from albumentations import HorizontalFlip, VerticalFlip
import numpy as np
import glob
import cv2
import os


from megvision.comm.tuple_functools import _pair


class CityScapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths, output_size=(512, 1024),
                 augmentation=False, super_reso=False, interpolation=cv2.INTER_LINEAR,
                 upscale_rate=4, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        assert len(image_paths) == len(mask_paths), "The length of list of image paths " \
                                                    "and masks are except to equal but " \
                                                    "got {} and {}".format(len(image_paths),len(mask_paths))
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.output_size = _pair(output_size)
        self.augmentation = augmentation
        self.super_reso = super_reso
        self.interpolation = interpolation
        self.upscale_rate = upscale_rate
        self.mean = mean
        self.std = std
        self.length = len(image_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        assert os.path.exists(image_path), "The image file {} is not exists".format(image_path)
        assert os.path.exists(mask_path), "The mask file {} is not exists".format(mask_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        if self.augmentation:
            aug_tasks = [
                ColorJitter(),
                ShiftScaleRotate(interpolation=self.interpolation, scale_limit=0.5),
                ChannelDropout(),
                ChannelShuffle(),
                HorizontalFlip(),
                VerticalFlip(),
                # PadIfNeeded(min_height=self.output_size[0], min_width=self.output_size[1])
            ]
            aug = Compose(aug_tasks)
            aug_data = aug(image=image, mask=mask)
            image = aug_data["image"]
            mask = aug_data["mask"]
        nor_resize = Compose([
            Resize(height=self.output_size[0], width=self.output_size[1], interpolation=self.interpolation),
            Normalize(mean=self.mean, std=self.std)
        ])
        if not self.super_reso:
            nor_resize_data = nor_resize(image=image, mask=mask)
            image = nor_resize_data["image"]
            mask = nor_resize_data["mask"]
        else:
            h,w = image.shape[:2]
            o_h, o_w = self.output_size
            if o_h * self.upscale_rate == h and o_w*self.upscale_rate==w:
                nor = Normalize(mean=self.mean, std=self.std)
                nor_data = nor(image=image, mask=mask)
                hr = nor_data["image"]
                mask = nor_data["mask"]
            else:
                o_h = o_h * self.upscale_rate
                o_w = o_w * self.upscale_rate
                nor_resize_hr = Compose([
                    Resize(width=o_w, height=o_h, interpolation=self.interpolation),
                    Normalize(mean=self.mean, std=self.std)
                ])
                data = nor_resize_hr(image=image, mask=mask)
                hr = data["image"]
                mask = data["mask"]
            hr = np.transpose(hr, axes=[2, 0, 1])
            nor_resize_data = nor_resize(image=image)
            image=nor_resize_data["image"]
        # mask = mask[mask<255]
        image = np.transpose(image, axes=[2, 0, 1])
        if self.super_reso:
            return image, hr, mask
        return image, mask

def get_paths(root, split="train"):
    img_data_dir = os.path.join(root, "leftImg8bit" ,split)
    mask_data_dir = os.path.join(root, "gtFine", split)
    img_paths = glob.glob(os.path.join(img_data_dir,"*","*.png"))
    mask_paths = []
    for path in img_paths:
        filename = os.path.basename(path)
        mask_filename = filename.replace("leftImg8bit", "gtFine_labelTrainIds")
        assert os.path.exists(os.path.join(mask_data_dir, mask_filename.split("_")[0], mask_filename)), os.path.join(mask_data_dir, mask_filename.split("_")[0], mask_filename)
        mask_paths.append(os.path.join(mask_data_dir, mask_filename.split("_")[0], mask_filename))
    return img_paths, mask_paths