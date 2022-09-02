# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:iostar
    author: 12718
    time: 2021/11/4 14:41
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from megengine.data.dataset import Dataset
from albumentations import Compose
from albumentations import ColorJitter, ShiftScaleRotate
from albumentations import GaussianBlur
from albumentations import HorizontalFlip, VerticalFlip, HueSaturationValue
from albumentations import Normalize
from albumentations import Resize
from PIL import Image
import skimage.io as imio
import numpy as np
import glob
import cv2
import os


from comm.tuple_functools import _pair

def get_paths(image_dir, mask_dir, image_suffix, mask_suffix):
    image_paths = glob.glob(os.path.join(image_dir, "*{}".format(image_suffix)))
    mask_paths = []
    for path in image_paths:
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        mask_path = os.path.join(mask_dir, name+mask_suffix)
        mask_paths.append(mask_path)
    return image_paths, mask_paths

def drive_get_paths(image_dir, mask_dir, image_suffix=".tif", mask_suffix=".gif"):
    image_paths = glob.glob(os.path.join(image_dir, "*{}".format(image_suffix)))
    mask_paths = []
    for path in image_paths:
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        id = name.split("_")[0]
        mask_name = "{}_manual1{}".format(id, mask_suffix)
        mask_path = os.path.join(mask_dir, mask_name)
        mask_paths.append(mask_path)
    return image_paths, mask_paths

class IOSTARDataset(Dataset):
    def __init__(self, image_paths, mask_paths, output_size, augmentation=False,
                 super_reso=False, green_channel=False, divide=False, interpolation=cv2.INTER_LINEAR,
                 upscale_rate=4, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), origin=False,
                 sssr=False):
        assert len(image_paths) == len(mask_paths), "Length of the image path lists must be equal," \
                                                    "but got len(image_paths)={} and len(mask_paths)={}".format(
            len(image_paths), len(mask_paths)
        )
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.green_channel = green_channel
        self.mean = mean
        self.std = std
        if self.green_channel:
            assert len(mean) == 1 and len(std) == 1, "If use the green channel of the image, " \
                                                     "please use the mean and std for the green channel," \
                                                     "except length of mean and std to 1 but got {} and {}".format(len(mean), len(std))
        self.interpolation = interpolation
        self.super_reso = super_reso
        self.upscale_rate = upscale_rate
        self.output_size = _pair(output_size)
        self.augmentation = augmentation
        self.length = len(image_paths)
        self.divide = divide
        self.origin_output = origin
        self.sssr = sssr

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        assert os.path.exists(image_path), "The image file {} is not exists.".format(image_path)
        assert os.path.exists(mask_path), "The mask file {} is not exists.".format(mask_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert image is not None
        basename = os.path.basename(mask_path)
        name, ext = os.path.splitext(basename)
        if ext == ".gif":
            mask = Image.open(mask_path)
            mask = np.array(mask)
        else:
            mask = cv2.imread(mask_path, 0)
        assert mask is not None
        if self.divide:
            mask = mask // 255
        height, width = image.shape[:2]
        if self.augmentation:
            aug_tasks = [
                ColorJitter(),
                # HueSaturationValue(),
                ShiftScaleRotate(rotate_limit=45),
                GaussianBlur(),
                HorizontalFlip(),
                VerticalFlip()
            ]
            aug_func = Compose(aug_tasks)
            aug_data = aug_func(image=image, mask=mask)
            image = aug_data["image"]
            mask = aug_data["mask"]
        normalize = Normalize(mean=self.mean, std=self.std)
        out_height = self.output_size[0]
        out_width = self.output_size[1]
        if self.super_reso and self.origin_output:
            hr = image.copy()
        elif self.super_reso and (out_width*self.upscale_rate != width
                                or out_height * self.upscale_rate != height):
            resize = Resize(height=out_height*self.upscale_rate, width=out_width*self.upscale_rate,
                            interpolation=cv2.INTER_CUBIC)
            re_data = resize(image=image, mask=mask)
            mask = re_data["mask"]
            hr = re_data["image"]
        elif self.super_reso:
            hr = image.copy()
        else:
            hr = None
        resize =  Resize(height=out_height, width=out_width,
                         interpolation=self.interpolation)
        re_data = resize(image=image, mask=mask)
        if self.sssr:
            mask = Resize(height=out_height * self.upscale_rate,
                          width=out_width * self.upscale_rate,
                          interpolation=self.interpolation)(image=image, mask=mask)["mask"]
        image = re_data["image"]

        # if hr is None and not self.sssr:
        mask = re_data["mask"]
        nor_data = normalize(image=image)
        image = nor_data["image"]
        if hr is not None:
            hr = normalize(image=hr)["image"]
        if self.green_channel:
            image = image[..., 1]
            if hr is not None:
                hr = hr[..., 1]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
            if hr is not None:
                hr = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            image = np.transpose(image, axes=[2, 0, 1])
            if hr is not None:
                hr = np.transpose(hr, axes=[2, 0, 1])
        if self.super_reso:
            return image, hr, mask
        else:
            return image, mask

def mask_preprocess(mask_dir, mask_suffix, av=False):
    mask_paths = glob.glob(os.path.join(mask_dir, "*"+mask_suffix))
    for mask_path in mask_paths:
        if av:
            data = cv2.imread(mask_path)
            mask = np.zeros(shape=(data.shape[0], data.shape[1], 3), dtype=np.uint8)
            mask1 = np.zeros(shape=(data.shape[0], data.shape[1], 3), dtype=np.uint8)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            mask = np.where(data == np.array((255, 0, 0)), np.array([1, 1, 1], dtype=np.uint8), mask)
            mask1 = np.where(data == np.array((0, 0, 255)), np.array([2, 2, 2], dtype=np.uint8), mask1)
            mask = mask + mask1
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask[mask < 2] = 0
            mask[mask > 2] = 1
        else:
            mask = cv2.imread(mask_path, 0)
            mask = mask // 255
        filename = os.path.basename(mask_path)
        name = os.path.splitext(filename)[0]
        imio.imsave(os.path.join(mask_dir, name+".png"), mask)

if __name__ == "__main__":
    image_dir = "D:/workspace/datasets/segmentation/IOSTAR/image/train"
    mask_dir = "D:/workspace/datasets/segmentation/IOSTAR/AV_GT/"
    # image_paths, mask_paths = get_paths(image_dir, mask_dir, ".jpg", "_AV.tif")
    mask_preprocess(mask_dir, mask_suffix="_AV.tif", av=True)
    # dataset = IOSTARDataset(image_paths, mask_paths, output_size=512, super_reso=True, upscale_rate=2)
    # image, hr, mask = dataset[0]
    # mask[mask < 2] = 0
    # print(hr.shape)
    # print(mask.shape)