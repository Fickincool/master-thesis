from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.fft as fft
import numpy as np
import torch.nn as nn
import torch
from scipy import io
from tqdm import tqdm


def train(model, train_loader, optimizer, lossf, epoch):
    """
    
    """

    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.squeeze().cuda()

        x, xf, recon, stripe = model(
            xc=data[0, :, :], weight_matrix=data[2, :, :], outlier_mask=data[1, :, :]
        )

        x = x.to(torch.double)
        target = data[0, :, :].to(torch.double)
        outlier_mask = data[1, :, :].to(torch.double)

        loss, tv, sim = lossf(x, target, outlier_mask, xf)
        loss.backward()
        # print(model.MGNN1.conv1.conv1.weight.grad)
        # print(model.MGNN1.conv1.conv1.conv_r.weight.grad)
        optimizer.step()
        # print(model.MGNN1.conv1.weight.grad.sum())
        if batch_idx % 2 == 0:
            print(
                "Train\t Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}\tTVLoss: {:.6f}\tSIM: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                    tv.item(),
                    sim.item(),
                )
                # loss.item(), 0, 0)
            )

    return model, x, recon, stripe
