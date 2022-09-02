# -*- coding:utf-8 -*-
# !/usr/bin/env python
"""
    filename:ssim
    author: 12718
    time: 2021/11/30 8:57
    tool: PyCharm
"""
import megengine as mge
import megengine.functional as F
import megengine.module as M

#Reference:https://github.com/jacke121/pytorch-ssim/blob/master/pytorch_ssim/__init__.py


def gaussian(window_size, sigma):
    """
    Compute one Gaussian distribution 1D-Tensor
    Parameters
    ----------
    window_size(int) :size of the window
    sigma (float): sigma

    Returns
    -------
        A 1D-tensor
    """
    _gaussian = mge.tensor([F.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
    return _gaussian / F.sum(_gaussian)


def create_window(window_size, channel):
    """
        Create the one window.
    Parameters
    ----------
    window_size (int): the size of the window
    channel (int): number of channels

    Returns
    -------
        4D-Tensor
    """
    _1D_window = F.expand_dims(gaussian(window_size, 1.5), 1) #w,1
    _2D_window = F.expand_dims(F.expand_dims(F.matmul(_1D_window, _1D_window.transpose()), 0), 0) #1x1xwxw
    window = F.broadcast_to(_2D_window, (channel, 1, 1, window_size, window_size))
    if channel == 1:
        window = F.squeeze(window, 1)
    return window

def _ssim(img1, img2, window, window_size, channel, size_average:bool = True):
    """

    Parameters
    ----------
    img1
    img2
    window
    window_size
    channel
    size_average

    Returns
    -------

    """
    mu1 = F.conv2d(img1, window, groups=channel, padding=window_size//2)
    mu2 = F.conv2d(img2, window, groups=channel, padding=window_size//2)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding= window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding= window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding= window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return F.mean(ssim_map)
    else:
        return F.mean(F.mean(F.mean(ssim_map, 1), 1), 1)

class SSIM(M.Module):
    def __init__(self, channel=1, window_size=11, size_average=True):
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.size_average = size_average
        super(SSIM, self).__init__()

    def forward(self, img1:mge.Tensor, img2:mge.Tensor):
        (_, ch, _, _) = img1.shape
        if ch == self.channel and self.window.dtype== img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, ch)

        window = mge.tensor(window, dtype=img1.dtype)
        return _ssim(img1, img2, window, self.window_size, ch, self.size_average)

class SSIMLoss(M.Module):
    def __init__(self, ch=3, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM(ch, window_size, size_average)

    def forward(self, output, target):
        loss = 1 - self.ssim(output, target)
        return loss


if __name__ == "__main__":
    window = create_window(3, 1)
    img1 = mge.random.normal(size=(1, 1, 256, 256))
    img2 = mge.random.normal(size=(1, 1, 256, 256))
    ssim_value = _ssim(img1, img2, window, 3, 1)
    print(ssim_value)