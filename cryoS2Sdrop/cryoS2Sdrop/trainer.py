import os
import yaml
from cryoS2Sdrop.dataloader import singleCET_dataset
from cryoS2Sdrop.model import Denoising_UNet

from torch.utils.data import DataLoader
import torch

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


class denoisingTrainer:
    def __init__(
        self, cet_path, subtomo_length, lr, n_features, p, tensorboard_logdir, loss_fn
    ):
        super().__init__()

        # Hardcoded
        self.loss_fn = loss_fn
        self.model = Denoising_UNet(self.loss_fn, lr, n_features, p)

        # model and training stuff
        self.cet_path = cet_path
        self.lr = lr
        self.subtomo_length = subtomo_length
        self.p = p
        self.n_features = n_features

        # logs
        self.tensorboard_logdir = tensorboard_logdir
        self.model_name = "s2sUNet"

        self.run_init_asserts()

        return

    def run_init_asserts(self):
        if self.subtomo_length % 32 != 0:
            raise ValueError(
                "Length of subtomograms must be a multiple of 32 to run the network."
            )
        return

    def train(
        self,
        batch_size,
        epochs,
        num_gpus,
        accelerator="gpu",
        strategy="ddp",
        transform=None,
    ):

        my_dataset = singleCET_dataset(
            self.cet_path,
            subtomo_length=self.subtomo_length,
            p=self.p,
            transform=transform,
        )

        print(
            "Size of dataset: %i, Steps per epoch: %i. \n"
            % (len(my_dataset), len(my_dataset) / (batch_size * num_gpus))
        )

        train_loader = DataLoader(
            my_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=aggregate_bernoulliSamples,
        )

        logger = pl_loggers.TensorBoardLogger(
            self.tensorboard_logdir, name="", default_hp_metric=False
        )

        early_stop_callback = EarlyStopping(
            monitor="hp/train_loss",
            min_delta=1e-4,
            patience=100,
            verbose=True,
            mode="min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks = [early_stop_callback, lr_monitor]

        trainer = Trainer(
            logger=logger,
            log_every_n_steps=1,
            gpus=num_gpus,
            max_epochs=epochs,
            enable_progress_bar=False,
            callbacks=callbacks,
            accelerator=accelerator,
            strategy=strategy,
        )

        trainer.fit(self.model, train_loader)

        if trainer.is_global_zero:
            #### Log additional hyperparameters #####
            hparams_file = os.path.join(
                self.tensorboard_logdir, "version_%i" % self.model.logger.version
            )
            hparams_file = os.path.join(hparams_file, "hparams.yaml")

            extra_hparams = {
                "transform": transform,
                "singleCET_dataset.Vmask_probability": my_dataset.Vmask_probability,
            }
            sdump = yaml.dump(extra_hparams)

            with open(hparams_file, "a") as fo:
                fo.write(sdump)

        return


def aggregate_bernoulliSamples(batch):
    """Flatten batch+bernoulli samples. Shape [B*M, C, S, S, S]

        Dataset returns [M, C, S, S, S] and dataloader returns [B, M, C, S, S, S].
        This function flattens the array in order to make a batch be the set of bernoulli samples of each of the B subtomos.
        """
    bernoulli_subtomo = torch.stack([b[0] for b in batch], axis=0).flatten(0, 1)
    target = torch.stack([b[1] for b in batch], axis=0).flatten(0, 1)
    bernoulli_mask = torch.stack([b[2] for b in batch], axis=0).flatten(0, 1)

    return bernoulli_subtomo, target, bernoulli_mask
