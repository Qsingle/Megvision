# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:iostar_train
    author: 12718
    time: 2021/11/4 16:29
    tool: PyCharm
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import megengine as mge
import megengine.autodiff as ad
import megengine.optimizer as optim
import megengine.functional as F
import megengine.functional.loss as mge_losses
import numpy as np
from megengine.data import DataLoader,RandomSampler,SequentialSampler
from megengine import amp
from megengine import dtr
from sklearn.model_selection import train_test_split
import megengine.distributed as dist
import argparse
import tqdm
import datetime
from tensorboardX.writer import SummaryWriter


from megvision.datasets.iostar import get_paths, IOSTARDataset, drive_get_paths
from megvision.comm.scheduler import PolyLRScheduler
from megvision.model.segmentation.deeplab import DeeplabV3Plus, DeeplabV3
from megvision.model.segmentation.unet import UNet
from megvision.model.segmentation.spunet import SpUNet
from megvision.model.segmentation import ESPNetV2_Seg
from megvision.model.segmentation.cs2net import CSNet
from megvision.model.segmentation.cenet import CENet
from megvision.model.segmentation import SAUNet
from megvision.model.segmentation.scsnet import SCSNet
from megvision.comm.metrics import SegmentationMetrcNumpy
from megvision.comm.loss.ssim import SSIMLoss
from megvision.comm.loss.dice import DiceLoss


best_iou = 0
global_step = 0
def train_val_step(model, train_dataloader, val_loader, optimizer:optim.Optimizer,
                   loss_fn, gm:ad.GradManager, epoch, args, log_path, lr_scheduler=None,
                   scaler=None, writer:SummaryWriter=None):

    global global_step
    bar = tqdm.tqdm(train_dataloader)
    losses = 0.0
    total = 0
    model.train()
    if args.amp:
        assert scaler is not None, "If use auto mix precision training, please set the scale"
    train_metrc = SegmentationMetrcNumpy(args.num_classes)

    for train_data in bar:
        if args.super_reso:
            x, hr, mask = train_data
            hr = mge.tensor(hr)
        else:
            x, mask = train_data
        x = mge.tensor(x, dtype="float32")
        total += int(x.shape[0])
        mask = mge.tensor(mask, dtype="int32")
        if args.amp:
            with amp.autocast():
                with gm:
                    pred = model(x)
                    if args.super_reso:
                        if isinstance(model, ESPNetV2_Seg):
                            pred1 = (pred[0], pred[2], pred[3], pred[4])
                            pred = pred[0]
                            loss = loss_fn(pred1, hr, mask) + F.loss.cross_entropy(pred[1], mask)
                        else:
                            loss = loss_fn(pred, hr, mask)
                    else:
                        if isinstance(model, ESPNetV2_Seg):
                            loss = loss_fn(pred[0], mask) + loss_fn(pred[1], mask)
                        else:
                            loss = loss_fn(pred, mask)
                    scaler.backward(gm, loss)
                optimizer.step()
                optimizer.clear_grad()
        else:
            with gm:
                pred = model(x)
                if args.super_reso:
                    if isinstance(model, ESPNetV2_Seg):
                        pred1 = (pred[0], pred[2], pred[3], pred[4])
                        pred = (pred[0], pred[1])
                        loss = loss_fn(pred1, hr, mask) + F.loss.cross_entropy(pred[1], mask)
                    else:
                        loss = loss_fn(pred, hr, mask)
                else:
                    if isinstance(model, ESPNetV2_Seg):
                        loss = loss_fn(pred[0], mask) + loss_fn(pred[1], mask)
                    else:
                        loss = loss_fn(pred, mask)
                gm.backward(loss)
            optimizer.step()
            optimizer.clear_grad()
        losses += loss.detach().item()

        if isinstance(model, ESPNetV2_Seg) and not args.super_reso:
            pred = sum(pred)
        if args.super_reso:
            pred = pred[0]
        if args.num_classes == 1:
            pred = F.sigmoid(pred)
        else:
            pred = F.softmax(pred, axis=1)
            pred = F.argmax(pred, axis=1)
        pred = F.flatten(pred)
        mask = F.flatten(mask)
        pred = pred.numpy()
        mask = mask.numpy()
        train_metrc.add_batch(pred, mask)
        results = train_metrc.evaluate()
        show_metrics = ["miou","mdice", "R", "P"]
        result_txt = ""
        if lr_scheduler is not None and isinstance(lr_scheduler, PolyLRScheduler):
            lr_scheduler.step()
        for k in show_metrics:
            if k in ["R", "P"]:
                if args.num_classes <= 2:
                    result_txt += "{}:{:.4f} ".format(k, results[k][1])
                else:
                    pass
            else:
                result_txt += "{}:{:.4f} ".format(k, results[k])
        if writer is not None:
            writer.add_scalar("train/loss", loss.numpy(), global_step=global_step)
            for k in show_metrics:
                if k in ["R", "P"]:
                    if args.num_classes <=2:
                        writer.add_scalar("train/{}".format(k), results[k][1], global_step=global_step)
                    else:
                        pass
                else:
                    writer.add_scalar("train{}".format(k), results[k], global_step=global_step)
            global_step += 1
        bar.set_description("[{}:{}] loss:{:.4f} {}".format(epoch+1,args.epochs, losses/ total,
                                                                          result_txt))

    model.eval()
    losses = 0.0
    total = 0
    val_metric = SegmentationMetrcNumpy(args.num_classes)
    for val_data in val_loader:
        if args.super_reso:
            x, hr, mask = val_data
            hr = mge.tensor(hr)
        else:
            x, mask = val_data
        x = mge.tensor(x, dtype="float32")
        total += int(x.shape[0])
        mask = mge.tensor(mask, dtype="int32")
        pred = model(x)
        if args.super_reso:
            loss = mge.functional.loss.cross_entropy(pred, mask)
        else:
            loss = loss_fn(pred, mask)
        if args.num_classes == 1:
            pred = F.sigmoid(pred)
        else:
            pred = F.softmax(pred, axis=1)
            pred = F.argmax(pred, axis=1)
        pred = F.flatten(pred)
        mask = F.flatten(mask)
        pred = pred.numpy()
        mask = mask.numpy()
        val_metric.add_batch(pred, mask)
        losses += loss.detach().item()
    results = val_metric.evaluate()
    show_metrics = ["miou", "mdice", "mean_p", "mean_r", "mean_acc", "P", "R", "ACC", "IoU"]
    result_txt = ""
    for k in show_metrics:
        if args.num_classes <= 2 and k in ["P", "R", "ACC", "IoU"]:
            result_txt += "{}:{:.4f} ".format(k, results[k][1])
        elif k in ["P", "R", "ACC", "IoU"]:
            pass
        else:
            result_txt += "{}:{:.4f} ".format(k, results[k])
    text = f"Validation: epoch:{epoch} loss:{losses/total} {result_txt}"
    print(text)
    with open(log_path, "w+", encoding="utf-8") as f:
        line = '{"epoch":%d %s}'% (epoch, result_txt)
        f.write(line)

    return results

# Distribution training
@dist.launcher
def dist_train(model, train_dataloader, val_dataloader, optimizer, loss_fn, gm:ad.GradManager, epochs, args, **kwargs):
    global best_iou
    if args.dtr:
        dtr.enable()
    dist.bcast_list_(model.tensors())
    gm.attach(model.parameters(), callbacks=[dist.make_allreduce_cb("sum")])
    rank = dist.get_rank()
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".json"
    log_file = os.path.join(log_dir, log_file)
    writer = kwargs.get("writer", None)
    with open(log_file, "w",encoding="utf-8") as f:
        f.write("{\n")
    print("Start training")
    if args.amp:
        scaler = amp.GradScaler()
        kwargs["scaler"] = scaler
    for epoch in range(epochs):
        results = train_val_step(model, train_dataloader, val_dataloader, optimizer, loss_fn, gm, epoch, args, log_file,
                                 **kwargs)
        if writer is not None:
            shows = ["miou", "mdice", "mean_p", "mean_r", "mean_acc"]
            for k in shows:
                writer.add_scalar("validation/{}".format(k), results[k], global_step=epoch)
        if rank == 0:
            if args.num_classes > 2:
                if best_iou < results["miou"]:
                    best_iou = results["miou"]
                    mge.save(model.state_dict(), os.path.join(ckpt_dir, "{}_best.pkl".format(args.model_name)))
            else:
                if best_iou < results["IoU"][1]:
                    best_iou = results["IoU"][1]
                    mge.save(model.state_dict(), os.path.join(ckpt_dir, "{}_best.pkl".format(args.model_name)))
            mge.save(model.state_dict(), os.path.join(ckpt_dir, "{}.pkl".format(args.model_name)))
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n}")

def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, gm:ad.GradManager, epochs, args, **kwargs):
    global best_iou
    print("Train on single card")
    if args.dtr:
        dtr.enable()
    ckpt_dir = args.ckpt_dir
    log_dir = args.log_dir
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".json"
    log_file = os.path.join(log_dir, log_file)
    if args.amp:
        scaler = amp.GradScaler()
        kwargs["scaler"] = scaler
    with open(log_file, "w", encoding="utf-8"):
        pass
    print("Start training")
    writer = kwargs.get("writer", None)
    for epoch in range(epochs):
        results = train_val_step(model, train_dataloader, val_dataloader, optimizer, loss_fn, gm, epoch, args, log_file, **kwargs)
        if writer is not None:
            shows = ["miou", "mdice", "mean_p", "mean_r", "mean_acc"]
            for k in shows:
                writer.add_scalar("validation/{}".format(k), results[k], global_step=epoch)
        if args.num_classes > 2:
            if best_iou < results["miou"]:
                best_iou = results["miou"]
                mge.save(model.state_dict(), os.path.join(ckpt_dir, "{}_best.pkl".format(args.model_name)))
        else:
            if best_iou < results["IoU"][1]:
                best_iou = results["IoU"][1]
                mge.save(model.state_dict(), os.path.join(ckpt_dir, "{}_best.pkl".format(args.model_name)))
        mge.save(model.state_dict(), os.path.join(ckpt_dir, "{}.pkl".format(args.model_name)))
    print(best_iou)

class SegDiceCELoss(mge.module.Module):
    def __init__(self, ce_loss=F.loss.cross_entropy, smooth=1.0):
        super(SegDiceCELoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = ce_loss
        self.smooth = smooth

    def forward(self, output, label):
        dice_loss = self.dice_loss(output, label)
        ce_loss = self.ce_loss(output, label)
        loss = dice_loss + ce_loss
        return loss

class SRLoss(mge.module.Module):
    def __init__(self, ch=3, window_size=11, size_average=True, alpha=0.85):
        super(SRLoss, self).__init__()
        self.ssim_loss = SSIMLoss(ch, window_size, size_average)
        self.mse_loss = F.loss.square_loss
        self.alpha = alpha

    def forward(self, lr, hr):
        ssim_loss = self.ssim_loss(lr, hr)
        mse_loss = self.mse_loss(lr, hr)
        return self.alpha*ssim_loss + (1-self.alpha)*mse_loss


class Seg_Super_SR(mge.module.Module):
    def __init__(self, alpha=1.0, beta=0.5,
                 seg_loss_fn=mge_losses.cross_entropy,
                 sr_loss_fn=mge_losses.square_loss):
        super(Seg_Super_SR, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.seg_loss_fn = seg_loss_fn
        self.sr_loss_fn = sr_loss_fn
        # self.fa_loss = FALoss()

    def forward(self, output, hr:mge.Tensor, target):
        pred = output[0]
        sr = output[1]

        # fa = output[2]

        # sr_qr = output[2]
        # qr_seg = output[2]
        # fusion = output[3]
        # features_seg = output[2]
        # features_sr = output[3]
        seg_loss = self.seg_loss_fn(pred, target)
        sr_loss = self.sr_loss_fn(sr, hr)
        # l1_loss = 0
        # for fe_sr, fe_seg in zip(features_sr, features_seg):
        #     l1_loss += F.loss.l1_loss(fe_seg, fe_sr)
        # qr_seg_loss = self.seg_loss_fn(qr_seg, target)
        # qr_sr_loss = self.sr_loss_fn(sr_qr, hr)
        # fa_loss = self.fa_loss(hr, fa)
        # fusion_loss = F.loss.binary_cross_entropy(fusion, F.broadcast_to(target, fusion.shape))
        # loss = self.alpha*seg_loss + self.beta*sr_loss + fa_loss
        loss = self.alpha*seg_loss + self.beta*sr_loss
        # loss = self.alpha * seg_loss + self.beta * sr_loss + self.alpha*qr_seg_loss
        return loss

def main(args):
    # args = parser.parse_args()
    model_name = args.model_name
    backbone = args.backbone
    num_workers = args.num_works
    batch_size = args.batch_size
    gpu = args.gpu
    pretrain = args.pretrain
    if gpu is not None:
        print("Using gpu {} to train".format(gpu))
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        mge.set_default_device("gpu{}".format(gpu))
    gpu_index = args.gpu_index
    if gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index
        print("Set CUDA_VISIBLE_DEVICES to {}".format(gpu_index))

    dist_traing = args.dist
    if dist_traing:
        print("Enable distribution train")
    init_lr = args.init_lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    num_classes = args.num_classes
    image_size = args.image_size
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    image_suffix = args.image_suffix
    mask_suffix = args.mask_suffix
    divide = args.divide
    dataset = args.dataset
    output_size = None
    origin_output = False
    sssr = args.sssr
    if dataset == "drive":
        divide = True
        origin_output = True
        if args.super_reso:
            output_size = [584, 565]
    # ckpt_dir = args.ckpt_dir
    # log_dir = args.log_dir
    super_reso = args.super_reso
    upscale_rate = args.upscale_rate
    paths, mask_paths = get_paths(image_dir, mask_dir, image_suffix=image_suffix, mask_suffix=mask_suffix)
    if dataset == "drive":
        paths, mask_paths = drive_get_paths(image_dir, mask_dir, image_suffix=image_suffix, mask_suffix=mask_suffix)
    elif dataset == "hrf":
        image_size = [584, 876]
        # image_size = [584, 836]
        # if model_name == "saunet":
        #     image_size = [1168, 1672]
        if super_reso and upscale_rate == 4:
            image_size = [292, 418]
            if model_name == "saunet":
                image_size = [584, 876]
    elif dataset == "prime":
        image_size = [648, 704]
        if super_reso and upscale_rate == 4:
            image_size = [324, 352]
            if model_name == "saunet":
                image_size = [648*2, 704*2]
    train_paths, val_paths, train_mask_paths, val_mask_paths = train_test_split(paths, mask_paths, random_state=0,
                                                                                test_size=0.2)
    if super_reso:
        print("Enable super resolution")
    train_dataset = IOSTARDataset(train_paths, train_mask_paths, output_size=image_size, divide=divide,
                                      super_reso=super_reso, augmentation=True, mean=[0.5, 0.5, 0.5],
                                      std=[0.5, 0.5, 0.5], upscale_rate=upscale_rate,
                                      green_channel=args.green, origin=origin_output,
                                      sssr=sssr
                                )
    val_dataset = IOSTARDataset(val_paths, val_mask_paths, output_size=image_size, divide=divide,
                                    super_reso=super_reso, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                                    upscale_rate=upscale_rate, green_channel=args.green, origin=origin_output,
                                    sssr=sssr
                               )
    train_sampler = RandomSampler(train_dataset, batch_size, drop_last=False)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, num_workers=num_workers)
    val_sampler = SequentialSampler(val_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, val_sampler, num_workers=num_workers)
    channels = 3
    print("Input size:{}".format(image_size))
    if args.green:
        channels = 1
    if model_name == "deeplabv3plus":
        model = DeeplabV3Plus(in_ch=channels, num_classes=num_classes, backbone=backbone,
                              super_reso=super_reso, zero_init_residual=False,
                              rcab=args.rcab, seb=args.seb, pretrained=pretrain,
                              upscale_rate=upscale_rate, sa=args.sa)

    elif model_name == "deeplabv3":
        model = DeeplabV3(in_ch=channels, num_classes=num_classes, backbone=backbone,
                          super_reso=super_reso, zero_init_residual=False,
                          pretrained=pretrain)
    elif model_name == "unet":
        model = UNet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, upscale_rate=upscale_rate,
                     output_size=output_size, sssr=sssr)
    elif model_name == "espnetv2":
        s = 0.5 if backbone == "espnetv2_s_0_5" else 1
        model = ESPNetV2_Seg(in_ch=channels, num_classes=num_classes, backbone=backbone, s=s, pretrained=True,
                             super_reso=super_reso, upscale_rate=upscale_rate)
    elif model_name == "saunet":
        model = SAUNet(in_ch=channels, num_classes=num_classes, block_size=7, keep_prob=0.82, super_reso=super_reso,
                       upscale_rate=upscale_rate, output_size=output_size)
    elif model_name == "spunet":
        model = SpUNet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, output_size=output_size)
    elif model_name == "csnet":
        model = CSNet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, output_size=output_size,
                      upscale_rate=upscale_rate)
    elif model_name == "cenet":
        model = CENet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, upscale_rate=upscale_rate,
                      out_size=output_size)
    elif model_name == "scsnet":
        model = SCSNet(in_ch=channels, num_classes=num_classes, super_reso=super_reso, out_size=output_size,
                       upscale_rate=upscale_rate)
    else:
        raise ValueError("Unknown model name {}".format(model_name))
    # optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    gm = ad.GradManager()
    if dist_traing:
        pass
    else:
        gm.attach(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    epochs = args.epochs
    print("train on {} samples, validation on {} samples".format(len(train_dataset), len(val_dataset)))
    # print(len(train_loader))
    lr_sche = PolyLRScheduler(optimizer, num_images=len(train_dataset), batch_size=batch_size, epochs=epochs)
    sr_name = "sr" if super_reso else "none_sr"
    writer = SummaryWriter(comment="{}_{}".format(model_name, sr_name))
    # seg_loss_fn = SegDiceCELoss()
    seg_loss_fn = F.loss.cross_entropy
    if num_classes == 1:
        # seg_loss_fn = SegDiceCELoss(seg_loss_fn=F.loss.binary_cross_entropy)
        seg_loss_fn = F.loss.binary_cross_entropy
    if super_reso:
        # sr_loss_fn = SRLoss(channels)
        sr_loss_fn = F.loss.square_loss
        loss_fn = Seg_Super_SR(sr_loss_fn=sr_loss_fn, seg_loss_fn=seg_loss_fn)
    else:
        loss_fn = seg_loss_fn
    if dist_traing:
        dist_train(model, train_loader, val_loader, optimizer, loss_fn, gm, epochs, args, lr_scheduler=lr_sche, writer=writer)
    else:
        train(model, train_loader, val_loader, optimizer, loss_fn, gm, epochs, args, lr_scheduler=lr_sche, writer=writer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Vessel segmentation train sample")
    parser.add_argument("--model_name", type=str, choices=["deeplabv3plus","deeplabv3", "unet", "espnetv2",
                                                           "saunet", "spunet", "csnet", "cenet", "scsnet"],
                        help="name of the model",
                        required=True)
    parser.add_argument("--backbone", type=str, choices=["resnet50", "resnet101", "resnest50",
                                                         "resnest200", "resnest14", "resnest26", "resnest101",
                                                         "espnetv2_s_0_5",
                                                         "espnetv2_s_1_0", "espnetv2_s_1_25",
                                                         "espnetv2_s_1_5", "espnetv2_s_2_0"]
                        ,help="backbone to extract features in some models",
                        default="resnet101")
    parser.add_argument("--green", action="store_true", default=False,
                        help="whether only use green channel")
    parser.add_argument("--num_works","-j", type=int, default=8,
                        help="number of workers to load data, default 8")
    parser.add_argument("--batch_size", "-bs", type=int, default=16,
                        help="size of each batch, default is 16")
    parser.add_argument("--gpu", type=int, default=None,
                        help="specific gpu index")
    parser.add_argument("--dist", action="store_true",
                        help="whether use distribution train")
    parser.add_argument("--epochs", type=int, default=300,
                        help="the number of train epochs")
    parser.add_argument("--init_lr", "-lr", default=0.01, type=float,
                        help="initial learning rate")
    parser.add_argument("--gpu_index", type=str, default=None)
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum used in optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay rate")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="path of directory that stored the images")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="path of directory that stored the masks")
    parser.add_argument("--image_suffix", type=str, default=".jpg",
                        help="suffix of the image file")
    parser.add_argument("--mask_suffix", type=str, default="_GT.tif",
                        help="suffix of the mask file")
    parser.add_argument("--num_classes", type=int, default=19,
                        help="number of classes")
    parser.add_argument("--image_size", type=int, default=512,
                        help="the image size used to train, default=512")
    parser.add_argument("--divide", action="store_true", default=False,
                        help="whether divide when load mask")
    parser.add_argument("--super_reso", action="store_true", default=False,
                        help="whether use super resolution")
    parser.add_argument("--upscale_rate", type=int, default=4,
                        help="upscale rate for super resolution")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpt",
                        help="path of directory to store the weights file")
    parser.add_argument("--log_dir", type=str, default="./log",
                        help="path of directory to store the log file")
    parser.add_argument("--dtr", action="store_true", default=False,
                        help="whether use dtr")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="whether use amp")
    parser.add_argument("--rcab", action="store_true", default=False,
                        help="whether use rcab in deeplabv3plus")
    parser.add_argument("--seb", action="store_true", default=False,
                        help="whether use rseb in deeplabv3plus")
    parser.add_argument("--pretrain", action="store_true", default=False,
                        help="whether use pretrained backbone")
    parser.add_argument("--sa", action="store_true", default=False,
                        help="whether use sa")
    parser.add_argument("--dataset", type=str, default="iostar",
                        help="name of the dataset")
    parser.add_argument("--sssr", action="store_true", default=False,
                        help="whether use sssr")
    args = parser.parse_args()
    if args.gpu_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
        print("Set CUDA_VISIBLE_DEVICES to {}".format(args.gpu_index))
    main(args)