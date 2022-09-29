# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (serpico team); adapted by Lorenz Lamm
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tomoSegmentPipeline.losses import Tversky_index


class DeepFinder_model(pl.LightningModule):
    def __init__(self, Ncl, loss_fn, lr, weight_decay, pretrain_type):
        super().__init__()
        self.Ncl = Ncl
        self.loss_fn = loss_fn
        self.pretrain_type = pretrain_type
        self.val_loss_per_epoch = []
        self.train_loss_per_epoch = []
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch_acc_train = []
        self.epoch_loss_train = []
        self.epoch_acc_val = []
        self.epoch_loss_val = []
        # Added only to assess membrane segmentation task
        self.val_dice1_per_epoch = []
        self.epoch_dice1_val = []

        self.save_hyperparameters()

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(48, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(48, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, (2, 2, 2), stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(64 + 48, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(48, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(48, 48, (2, 2, 2), stride=2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(48 + 32, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, self.Ncl, (1, 1, 1), padding=0),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x_high = self.layer1(x)
        mid = self.layer2(x_high)
        x = self.layer3(mid)
        x = torch.cat((mid, x), dim=1)
        x = self.layer4(x)
        x = torch.cat((x_high, x), dim=1)
        x = self.layer5(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        factor = 0.1

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    "min",
                    verbose=True,
                    patience=15,
                    min_lr=1e-7,
                    factor=factor,
                ),
                "monitor": "hp/train_loss_epoch",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def training_step(self, batch, batch_idx):
        batch_data, batch_target = batch
        pred = self(batch_data)
        loss = self.loss_fn(pred, batch_target)
        train_acc = torch.mean((pred.argmax(1) == batch_target.argmax(1)) * 1.0)

        self.log(
            "hp/train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "hp/train_acc",
            train_acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        batch_data, batch_target = batch
        pred = self(batch_data)
        loss = self.loss_fn(pred, batch_target)
        val_acc = torch.mean((pred.argmax(1) == batch_target.argmax(1)) * 1.0)

        self.log("hp/val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("hp/val_acc", val_acc, on_epoch=True, prog_bar=True, sync_dist=True)

        # Membrane segmentation assessment:
        # Only valid for Tversky1_loss
        # self.log("hp/val_dice", 1-loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # valid when loss is Tversky_loss
        dice1 = Tversky_index(pred, batch_target)[1]
        self.log("hp/val_dice", dice1, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        print(self.hparams)


class PretextDeepFinder_model(pl.LightningModule):
    # Used for pretext regression task
    def __init__(self, loss_fn, lr, weight_decay):
        super().__init__()
        self.loss_fn = loss_fn
        self.pretrain_type = None
        self.val_loss_per_epoch = []
        self.train_loss_per_epoch = []
        # self.loss_fn.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch_acc_train = []
        self.epoch_loss_train = []
        self.epoch_acc_val = []
        self.epoch_loss_val = []
        self.save_hyperparameters()

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(48, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(48, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 64, (2, 2, 2), stride=2),
            nn.Conv3d(64, 64, (3, 3, 3), padding=1),
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(64 + 48, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(48, 48, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(48, 48, (2, 2, 2), stride=2),
            nn.Conv3d(48, 48, (3, 3, 3), padding=1),
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(48 + 32, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 32, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, (1, 1, 1), padding=0),
        )

    def forward(self, x):
        x_high = self.layer1(x)
        mid = self.layer2(x_high)
        x = self.layer3(mid)
        x = torch.cat((mid, x), dim=1)
        x = self.layer4(x)
        x = torch.cat((x_high, x), dim=1)
        x = self.layer5(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_data, batch_target = batch
        # batch_data = batch_data.to(self.device)
        # batch_target = batch_target.to(self.device)
        pred = self(batch_data)
        loss = self.loss_fn(pred, batch_target)
        self.log("hp/train loss", loss)
        self.epoch_loss_train.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_data, batch_target = batch
        # batch_data = batch_data.to(self.device)
        # batch_target = batch_target.to(self.device)
        pred = self(batch_data)
        loss = self.loss_fn(pred, batch_target)
        self.log("hp/val loss", loss)
        self.epoch_loss_val.append(loss)
        return loss

    def on_epoch_end(self):

        if len(self.epoch_loss_train) > 0:
            train_loss_epoch = torch.mean(torch.Tensor(self.epoch_loss_train))
            self.log("hp/train_loss_epoch", train_loss_epoch)
            self.log(
                "hp/train_acc_epoch", torch.mean(torch.Tensor(self.epoch_acc_train))
            )
            self.train_loss_per_epoch.append(train_loss_epoch)

        if len(self.epoch_loss_val) > 0:
            val_loss_epoch = torch.mean(torch.Tensor(self.epoch_loss_val))
            self.log("hp/val_loss_epoch", torch.mean(val_loss_epoch))
            self.log("hp/val_acc_epoch", torch.mean(torch.Tensor(self.epoch_acc_val)))
            self.val_loss_per_epoch.append(val_loss_epoch)

        self.epoch_acc_train = []
        self.epoch_loss_train = []
        self.epoch_acc_val = []
        self.epoch_loss_val = []

    def on_train_start(self):
        self.logger.log_hyperparams(
            self.hparams, {"hp/train loss": 0, "hp/mean_pred": 0}
        )
        print(self.hparams)
