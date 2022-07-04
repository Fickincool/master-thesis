import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cryoS2Sdrop.partialconv3d import PartialConv3d

class Denoising_UNet(pl.LightningModule):
    def __init__(self, loss_fn, lr, n_samples, n_features, p):
        super().__init__()
        self.loss_fn = loss_fn
        self.lr = lr
        self.n_features = n_features
        self.n_samples = n_samples
        self.p = p
        self.save_hyperparameters()

        # Encoder blocks
        self.EB1 = PartialConv3d(n_samples, self.n_features, kernel_size=3, padding=1)
        self.EB2 = self.encoder_block()
        self.EB3 = self.encoder_block()
        self.EB4 = self.encoder_block()
        self.EB5 = self.encoder_block()
        self.EB6 = self.encoder_block()
        self.EB_bottom = self.encoder_block_bottom()

        # Upsampling
        self.up65 = nn.Upsample(scale_factor=2)
        self.up54 = nn.Upsample(scale_factor=2)
        self.up43 = nn.Upsample(scale_factor=2)
        self.up32 = nn.Upsample(scale_factor=2)
        self.up21 = nn.Upsample(scale_factor=2)

        # decoder blocks
        self.DB5 = self.decoder_block(2*n_features, 2*n_features)
        self.DB4 = self.decoder_block(3*n_features, 2*n_features)
        self.DB3 = self.decoder_block(3*n_features, 2*n_features)
        self.DB2 = self.decoder_block(3*n_features, 2*n_features)
        self.DB1 = self.decoder_block_top()

        return

    def forward(self, x:torch.Tensor):
        "Input tensor of shape [batch_size, sample_size, tomo_side, tomo_side, tomo_side]"
        ##### ENCODER #####
        e1 = self.EB1(x) # no downsampling, n_features = 48
        e2 = self.EB2(e1) # downsample 1/2
        e3 = self.EB3(e2) # 1/4
        e4 = self.EB4(e3) # 1/8
        e5 = self.EB5(e4) # 1/16
        e6 = self.EB6(e5) # 1/32
        e_bottom = self.EB_bottom(e6) # 1/32, n_features = 48

        ##### DECODER #####
        d5 = self.up65(e_bottom) # 1/16
        d5 = torch.concat([d5, e5], axis=1) # 1/16, n_freatures = 96
        d5 = self.DB5(d5) # 1/16

        d4 = self.up54(d5) # 1/8
        d4 = torch.concat([d4, e4], axis=1) # 1/8 n_features = 144
        d4 = self.DB4(d4) # 1/8 n_features = 96

        d3 = self.up43(d4) # 1/4
        d3 = torch.concat([d3, e3], axis=1) # 1/4
        d3 = self.DB3(d3) # 1/4

        d2 = self.up32(d3) # 1/2
        d2 = torch.concat([d2, e2], axis=1) # 1/2
        d2 = self.DB2(d2) # 1/2

        d1 = self.up21(d2)
        d1 = torch.concat([d1, x], axis=1)
        x = self.DB1(d1)

        return x

    def encoder_block(self):
        layer = nn.Sequential(
            PartialConv3d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        return layer
    
    def encoder_block_bottom(self):
        layer = nn.Sequential(
            PartialConv3d(self.n_features, self.n_features, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        return layer

    def decoder_block(self, n_features_in, n_features_out):
        layer = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv3d(n_features_in, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(self.p),
            nn.Conv3d(n_features_out, n_features_out, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        return layer
    
    def decoder_block_top(self):
        layer = nn.Sequential(
            nn.Dropout(self.p),
            nn.Conv3d(2*self.n_features+self.n_samples, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(self.p),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(self.p),
            nn.Conv3d(32, self.n_samples, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        return layer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        factor = 0.1

        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=10, min_lr=1e-7, factor=factor),
            "monitor": "hp/train_loss_epoch",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def training_step(self, batch):
        bernoulli_subtomo, target, bernoulli_mask = batch
        pred = (1-bernoulli_mask)*self(bernoulli_subtomo)
        loss = self.loss_fn(pred, target)

        self.log("hp/train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss