# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:iostar_evaluate
    author: 12718
    time: 2021/11/4 18:53
    tool: PyCharm
"""
import glob
import pathlib

import megengine as mge
from albumentations import Normalize
from albumentations import Resize
import cv2
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np
import os
import pathlib
from PIL import Image


from megvision.comm.metrics.segmentation_metrics import SegmentationMetrcNumpy
from megvision.model.segmentation import DeeplabV3Plus, UNet, DeeplabV3, ESPNetV2_Seg,SAUNet
from megvision.model.segmentation import CENet
from megvision.model.segmentation.spunet import SpUNet
from megvision.model.segmentation.cs2net import CSNet
from megvision.model.segmentation.scsnet import SCSNet
from megvision.model.segmentation.pfseg import PFSeg
from megvision.datasets.iostar import get_paths, drive_get_paths
from megvision.comm.tuple_functools import _pair

def numeric_score(pred, gt):
    FP = np.float64(np.sum((pred == 1) & (gt == 0)))
    FN = np.float64(np.sum((pred == 0) & (gt == 1)))
    TP = np.float64(np.sum((pred == 1) & (gt == 1)))
    TN = np.float64(np.sum((pred == 0) & (gt == 0)))
    return FP, FN, TP, TN

def get_path_prime(image_dir, mask_dir, image_suffix=".tif", mask_suffix=".png"):
    image_paths = glob.glob((os.path.join(image_dir, "*{}".format(image_suffix))))
    mask_paths = []
    for path in image_paths:
        basename = os.path.basename(path)
        filename = os.path.splitext(basename)[0]
        filename = filename.replace("Img", "Label") + mask_suffix
        mask_paths.append(os.path.join(mask_dir, filename))
    return image_paths, mask_paths


def main():
    args = parser.parse_args()
    model_name = args.model_name
    backbone = args.backbone
    gpu = args.gpu
    if gpu is not None:
        print("Using gpu {} to prediction".format(gpu))
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        mge.set_default_device("gpu{}".format(gpu))


    num_classes = args.num_classes
    image_size = args.image_size
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    image_suffix = args.image_suffix
    mask_suffix = args.mask_suffix
    divide = args.divide
    dataset = args.dataset
    out_size = None
    super_reso = args.super_reso
    upscale_rate = args.upscale_rate
    output_size = _pair(image_size)
    image_paths, mask_paths = get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
    if dataset == "drive":
        image_paths, mask_paths = drive_get_paths(image_dir, mask_dir, image_suffix, mask_suffix)
        out_size = [585, 564]
    elif dataset == "hrf":
        output_size = [584, 876]
        # output_size = [584, 836]
        # if model_name == "saunet":
        #     output_size = [1168, 1672]
        if super_reso and upscale_rate == 4:
            output_size = [292, 418]
            if model_name == "saunet":
                output_size = [584, 876]
    elif dataset == "prime":
        output_size = [648, 704]
        if super_reso and upscale_rate == 4:
            output_size = [324, 352]
    weigths = args.weights
    assert os.path.exists(weigths), "weights file {} not exists.".format(weigths)
    channels = 3
    sssr = args.sssr


    if args.green:
        channels = 1
    if model_name == "deeplabv3plus":
        model = DeeplabV3Plus(in_ch=channels, num_classes=num_classes, backbone=backbone,
                              super_reso=super_reso, zero_init_residual=False,
                              rcab=args.rcab, seb=args.seb, upscale_rate=upscale_rate, sa=args.sa)

    elif model_name == "deeplabv3":
        model = DeeplabV3(in_ch=channels, num_classes=num_classes, backbone=backbone,
                          super_reso=super_reso, zero_init_residual=False,
                          )
    elif model_name == "unet":
        model = UNet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, upscale_rate=upscale_rate,
                     output_size=out_size, sssr=sssr)
    elif model_name == "espnetv2":
        s = 0.5 if backbone == "espnetv2_s_0_5" else 1
        model = ESPNetV2_Seg(in_ch=channels, num_classes=num_classes, backbone=backbone, s=s, pretrained=True,
                             super_reso=super_reso, upscale_rate=upscale_rate)
    elif model_name == "saunet":
        model = SAUNet(in_ch=channels, num_classes=num_classes, super_reso=super_reso,
                       upscale_rate=upscale_rate, keep_prob=0.8, block_size=7,
                       output_size=out_size)
    elif model_name == "spunet":
        model = SpUNet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, output_size=out_size)
    elif model_name == "csnet":
        model = CSNet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, output_size=out_size,
                      upscale_rate=upscale_rate)
    elif model_name == "cenet":
        model = CENet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, out_size=out_size,
                      upscale_rate=upscale_rate)
    elif model_name == "scsnet":
        model = SCSNet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, out_size=out_size,
                       upscale_rate=upscale_rate)
    elif model_name == "pfseg":
        model = PFSeg(in_ch=channels, num_classes=num_classes)
    else:
        raise ValueError("Unknown model name {}".format(model_name))
    print(weigths)
    with open(weigths, "rb") as f:
        state = mge.load(f, map_location=lambda dev: 'cpu0')
    model.load_state_dict(state)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalize = Normalize(mean=mean, std=std)
    outpu_dir = args.output_dir
    if not os.path.exists(outpu_dir):
        os.makedirs(outpu_dir)
    out_height = output_size[0]
    out_width = output_size[1]
    # if super_reso:
    #     out_width = out_width * upscale_rate
    #     out_height = out_height * upscale_rate
    #     if out_size is not None:
    #         out_height = out_size[0]
    #         out_width = out_size[1]
    print(out_width, out_height)
    metric = SegmentationMetrcNumpy(num_classes=num_classes)
    otsu_metric = SegmentationMetrcNumpy(num_classes=num_classes)
    model.eval()
    spe = []
    acc = []
    sen = []
    aucs = []
    ssims = []
    msssims = []
    uqis = []
    psnrs = []
    psnrbs = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        hr_gt = image
        ext = os.path.splitext(filename)[1]
        if dataset == "drive" or ext == ".gif":
            mask = np.array(Image.open(mask_path))
        if divide:
            mask = mask // 255
        height, width = image.shape[:2]
        hr = image
        resize = Resize(height=out_height, width=out_width)
        re_data = resize(image=image)
        image = re_data["image"]
        nor_data = normalize(image=image)
        image = nor_data["image"]
        if (out_width * upscale_rate != width or out_height*upscale_rate != height) and super_reso:
            hr = Resize(out_height*upscale_rate, out_width*upscale_rate, interpolation=cv2.INTER_CUBIC)(image=hr)["image"]
            cv2.imwrite(os.path.join(outpu_dir, os.path.splitext(filename)[0] + "_hr_gt.png"), cv2.cvtColor(hr, cv2.COLOR_RGB2BGR))
        if hr is not None:
            hr = normalize(image=hr)["image"]
        guidance = None
        if isinstance(model, PFSeg):
            crop_width = out_width // 2
            crop_height = out_height // 2
            c_x = hr.shape[1] // 2
            c_y = hr.shape[0] // 2
            guidance = hr[c_y-crop_height//2:c_y+crop_height//2, c_x-crop_width//2:c_x+crop_width//2, :]
            guidance = np.array(guidance, dtype=np.float32)
            guidance = np.transpose(guidance, axes=[2, 0, 1])
        if args.green:
            image = image[..., 1]
            if hr is not None:
                hr = hr[..., 1]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
            if hr is not None:
                hr = np.expand_dims(hr, axis=0)
        elif image.ndim == 3:
            image = np.transpose(image, axes=[2, 0, 1])
            if hr is not None:
                hr = np.transpose(hr, axes=[2, 0, 1])
        image = mge.tensor(image, dtype="float32")
        image = mge.functional.expand_dims(image, axis=0)
        if guidance is None:
            pred = model(image)
        else:
            guidance = mge.tensor(guidance)
            guidance = mge.functional.expand_dims(guidance, axis=0)
            pred = model(image, guidance)
        sr = None
        qr_out = None
        qr_hr = None
        fu = None

        if isinstance(pred, tuple):
            if len(pred) > 3:
                pred, sr, qr_out, fu = pred
            elif len(pred) > 2:
                pred, sr, fu = pred
            else:
                pred, sr = pred
        print(pred.shape)
        if num_classes < 2:
            pred = mge.functional.sigmoid(pred)
            pred = pred.numpy()
            pred = np.where(pred >= 0.5, 1, 0)
            pred = mge.tensor(pred)
        else:
            pred = mge.functional.softmax(pred, axis=1)
            pred = mge.functional.argmax(pred, axis=1)
        pred = pred.numpy().squeeze()
        if fu is not None:
            # fu_seg = fu.numpy().squeeze()

            fu = fu*255
            fu_seg = fu.numpy().squeeze()
            fu_seg = np.uint8(fu_seg)
            heat = cv2.applyColorMap(fu_seg, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(outpu_dir, os.path.splitext(filename)[0] + "_fu.png"), fu_seg)
            cv2.imwrite(os.path.join(outpu_dir, os.path.splitext(filename)[0] + "_heat.png"), heat)
        if sr is not None:
            sr = sr.numpy().squeeze()
            sr = np.transpose(sr, axes=[1, 2, 0])
            sr = (sr - sr.min())/(sr.max() - sr.min() + 1e-9)
            sr = sr * 255
            sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(outpu_dir, os.path.splitext(filename)[0]+"_sr.png"), sr)

        if qr_out is not None:
            qr_out = mge.functional.argmax(qr_out, axis=1)
            qr_out = qr_out.numpy().squeeze() * 255
            cv2.imwrite(os.path.join(outpu_dir, os.path.splitext(filename)[0]+"_qr_out.png"), qr_out)
        if qr_hr is not None:
            qr_hr = qr_hr.numpy().squeeze()
            qr_hr = np.transpose(qr_hr, axes=[1, 2, 0])
            qr_hr = (qr_hr - qr_hr.min()) / (qr_hr.max() - qr_hr.min() + 1e-9) * 255
            qr_hr = np.uint8(qr_hr)
            qr_hr = cv2.cvtColor(qr_hr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(outpu_dir, os.path.splitext(filename)[0] + "_qr_hr.png"), qr_hr)
        output_height, output_width = pred.shape
        if super_reso and (output_width != width or output_height != height):
            pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        elif not super_reso and (output_width != width or output_height != height):
            pred = cv2.resize(pred, (width, height), interpolation=cv2.INTER_NEAREST)
        metric.add_batch(pred, mask)
        FP, FN, TP, TN = numeric_score(pred, mask)
        auc = roc_auc_score(mask.ravel(), pred.ravel())
        spe.append(TN/(TN + FP))
        acc.append((TP+TN)/(FN+FP+TP+TN))
        sen.append(TP/(TP+FN))
        aucs.append(auc)
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9) * 255
        cv2.imwrite(os.path.join(outpu_dir, os.path.splitext(filename)[0]+".png"), pred)
        thresh, pred = cv2.threshold(np.uint8(pred), 0, 255, cv2.THRESH_OTSU)
        pred = pred // 255
        otsu_metric.add_batch(pred, mask)

    results = metric.evaluate()
    show_metrics = ["miou", "mdice", "mean_p", "mean_r", "mean_acc", "mean_sp"]
    result_txt = ""
    for k in show_metrics:
        result_txt += "{}:{:.4f} ".format(k, results[k])
    others_metrics = ["SP", "P", "R", "IoU", "dice", "ACC"]
    for key in others_metrics:
        if num_classes <= 2:
            result_txt += " {}:{} ".format(key, results[key][1])
        else:
            result_txt += " {}:{} ".format(key, results[key])
    text = f"{result_txt} auc:{np.mean(aucs)}"
    print(text)

    print("spe:{} sen:{} acc:{} auc:{}".format(np.mean(spe), np.mean(sen), np.mean(acc), np.mean(aucs)))

    print("After OTSU:")
    results = otsu_metric.evaluate()
    show_metrics = ["miou", "mdice", "mean_p", "mean_r", "mean_acc", "mean_sp"]
    result_txt = ""
    for k in show_metrics:
        result_txt += "{}:{:.4f} ".format(k, results[k])
    others_metrics = ["SP", "P", "R", "IoU", "dice", "ACC", "gmean"]
    for key in others_metrics:
        if num_classes <= 2:
            result_txt += " {}:{} ".format(key, results[key][1])
        else:
            result_txt += " {}:{} ".format(key, results[key])
    text = f"{result_txt}"
    print(text)
    if len(ssims) > 0:
        print(f"psnr:{np.mean(psnrs)} psnrb:{np.mean(psnrbs)} ssim:{np.mean(ssims)} msssim:{np.mean(msssims)} uqi:{np.mean(uqis)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vessel segmentation evaluate sample")
    parser.add_argument("--model_name", type=str, choices=["deeplabv3plus", "deeplabv3", "unet", "espnetv2", "saunet",
                                                           "spunet", "csnet", "cenet", "scsnet", "pfseg"],
                        help="name of the model",
                        required=True)
    parser.add_argument("--backbone", type=str, choices=["resnet50", "resnet101", "resnest50",
                                                         "resnest200", "resnest14", "resnest26", "resnest101",
                                                         "espnetv2_s_0_5",
                                                         "espnetv2_s_1_0", "espnetv2_s_1_25",
                                                         "espnetv2_s_1_5", "espnetv2_s_2_0"]
                        , help="backbone to extract features in some models",
                        default="resnet101")
    parser.add_argument("--green", action="store_true", default=False,
                        help="whether only use green channel")
    parser.add_argument("--gpu", type=int, default=None,
                        help="specific gpu index")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="path of directory that stored the images")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="path of directory that stored the masks")
    parser.add_argument("--image_suffix", type=str, default=".jpg",
                        help="suffix of the image file")
    parser.add_argument("--mask_suffix", type=str, default="_GT.tif",
                        help="suffix of the mask file")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="number of classes")
    parser.add_argument("--image_size", type=int, default=512,
                        help="the image size used to train, default=512")
    parser.add_argument("--divide", action="store_true", default=False,
                        help="whether divide when load mask")
    parser.add_argument("--super_reso", action="store_true", default=False,
                        help="whether use super resolution")
    parser.add_argument("--upscale_rate", type=int, default=4,
                        help="upscale rate for super resolution")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="whether use amp")
    parser.add_argument("--rcab", action="store_true", default=False,
                        help="whether use rcab in deeplabv3plus")
    parser.add_argument("--sa", action="store_true", default=False,
                        help="whether use rcab in deeplabv3plus")
    parser.add_argument("--seb", action="store_true", default=False,
                        help="whether use rseb in deeplabv3plus")
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, default="./output")
    parser.add_argument("--dataset", type=str, default="iostar")
    parser.add_argument("--sssr", action="store_true", default=False,
                        help="Whether use sssr")
    main()