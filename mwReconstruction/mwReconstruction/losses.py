from pytorch_msssim import ms_ssim, ssim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.fft as fft
import numpy as np
import torch.nn as nn
import torch
from scipy import io


class TotalVariationRegularizationLoss(nn.Module):
    def __init__(self, total_variation_weight):
        super().__init__()
        self.total_variation_weight = total_variation_weight

    def forward(self, input):
        loss = torch.sum(torch.abs(input[:, :, :-1] - input[:, :, 1:]))
        # + torch.sum(torch.abs(input[:, :-1, :] - input[:, 1:, :]))
        loss = loss * self.total_variation_weight
        return loss


class HessianLoss(nn.Module):
    def __init__(self, hessian_weight, path):
        super().__init__()
        self.hessian_weight = hessian_weight
        self.Hessianxx = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(19, 19), stride=1, bias=False
        )
        self.Hessianyy = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=(19, 19), stride=1, bias=False
        )
        f = io.loadmat(path + "DGaussxx.mat")["DGaussxx"]
        kernel_xx, kernel_yy = f, f.T
        self.Hessianxx.weight.data = (
            torch.from_numpy(kernel_yy.astype("float32")).unsqueeze(0).unsqueeze(0)
        )
        self.Hessianyy.weight.data = (
            torch.from_numpy(kernel_xx.astype("float32")).unsqueeze(0).unsqueeze(0)
        )
        self.Hessianxx.weight.requires_grad = False
        self.Hessianyy.weight.requires_grad = False

    def forward(self, x, target):
        return (self.Hessianxx(x).abs()).sum() + (
            self.Hessianyy(x - target).abs()
        ).sum()


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # path = os.path.join(PARENT_PATH, 'destripe/Data/')
        self.lr = nn.MSELoss(reduction="sum")
        self.li = nn.MSELoss(reduction="sum")
        self.TotalVariationRegularizationLoss = TotalVariationRegularizationLoss(1)
        # self.HessianLoss = HessianLoss(1, path)
        self.l = ssim

    def forward(self, x, target, hier_mask, xf):
        tv = self.TotalVariationRegularizationLoss(x.unsqueeze(1))
        # hessian = self.HessianLoss(x.unsqueeze(0), target.unsqueeze(0))
        # sim = self.l(
        #     x.squeeze().unsqueeze(0).unsqueeze(0),
        #     torch.log10(self.original).squeeze().unsqueeze(0).unsqueeze(0),
        # )
        target = fft.fftshift(fft.fft2(target))

        mse1 = self.lr(xf.real, target.real)
        mse2 = self.li(xf.imag, target.imag)

        # xx = x[:, :-1, :] - x[:, 1:, :]
        # torch.abs(xx[:, :-1, :] - xx[:, 1:, :])

        return mse1 + mse2 + tv, tv, mse1
