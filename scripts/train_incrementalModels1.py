from tomoSegmentPipeline import dataloader as dl
from tomoSegmentPipeline.utils import setup
from tomoSegmentPipeline.dataloader import (
    to_categorical,
    transpose_to_channels_first,
    tomoSegment_dummyDataset,
    tomoSegment_dataset,
)
from tomoSegmentPipeline.training import Train
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor


import os

PARENT_PATH = setup.PARENT_PATH

import torch

val_tomos = ["tomo32", "tomo10"]
concat_val_ids = sorted([s.replace("tomo", "") for s in val_tomos])
concat_val_ids = "-".join(concat_val_ids)

test_tomos = ["tomo38", "tomo04"]
concat_test_ids = sorted([s.replace("tomo", "") for s in test_tomos])
concat_test_ids = "-".join(concat_test_ids)


train_tomos = ["tomo02", "tomo03", "tomo17"]
concat_train_ids = sorted([s.replace("tomo", "") for s in train_tomos])
concat_train_ids = "-".join(concat_train_ids)

for i in range(5):
    # in 2 days, at most 3 complete rounds of incremental models seem to be able to finish
    for input_type in ["rawCET", "cryoCARE", "isoNET", "cryoCARE+isoNET"]:
        for nPatches in range(4, 34, 4):

            tb_logdir = os.path.join(
                PARENT_PATH,
                "data/model_logs/incremental_models/logs/BaselineModel/%s/train%s/nPatches_%i"
                % (input_type, concat_train_ids, nPatches),
            )
            model_name = "Baseline_" + input_type

            epochs = 1000
            Ncl = 2
            dim_in = 84
            lr = 1e-4
            weight_decay = 0
            Lrnd = 18
            augment_data = True
            batch_size = 22
            pretrained_model = None

            trainer = Train(
                Ncl=Ncl,
                dim_in=dim_in,
                lr=lr,
                weight_decay=weight_decay,
                Lrnd=Lrnd,
                tensorboard_logdir=tb_logdir,
                model_name=model_name,
                augment_data=augment_data,
                batch_size=batch_size,
                epochs=epochs,
                pretrained_model=pretrained_model,
            )

            early_stop_callback = EarlyStopping(
                monitor="hp/val_loss",
                min_delta=1e-4,
                patience=100,
                verbose=True,
                mode="min",
            )

            lr_monitor = LearningRateMonitor(logging_interval="step")
            callbacks = [early_stop_callback, lr_monitor]

            trainer.launch(
                train_tomos,
                val_tomos,
                input_type=input_type,
                num_gpus=2,
                accelerator="ddp",
                nPatches_training=nPatches,
                num_workers=1,
                train_callbacks=callbacks,
            )

            del trainer
