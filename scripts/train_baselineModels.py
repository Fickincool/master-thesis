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


training_schedule = [["tomo02", "tomo03", "tomo17"], ["tomo02"], ["tomo03"], ["tomo17"]]
# training_schedule = [['tomo02'], ['tomo03'], ['tomo17']]

for input_type in ["isoNET", "cryoCARE+isoNET", "cryoCARE", "rawCET"]:
    for train_tomos in training_schedule:
        concat_train_ids = sorted([s.replace("tomo", "") for s in train_tomos])
        concat_train_ids = "-".join(concat_train_ids)

        chkpnt = None

        if len(train_tomos) == 1:
            tb_logdir = os.path.join(
                PARENT_PATH,
                "data/model_logs/models_2/logs/LowBaselineModel/%s/train%s"
                % (input_type, concat_train_ids),
            )
            model_name = "LowBaseline"

            # chkpnt = os.path.join(tb_logdir, 'version_4/checkpoints/epoch=799-step=1599.ckpt')
            # epochs += 200

        elif len(train_tomos) == 3:
            tb_logdir = os.path.join(
                PARENT_PATH,
                "data/model_logs/models_2/logs/BaselineModel/%s/train%s"
                % (input_type, concat_train_ids),
            )
            model_name = "Baseline"

            # chkpnt = os.path.join(tb_logdir, 'version_10/checkpoints/epoch=699-step=3499.ckpt')
            # epochs += 200

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
            num_gpus=3,
            accelerator="ddp",
            num_workers=1,
            resume_from_checkpoint=chkpnt,
            train_callbacks=callbacks,
        )


# #################### TEST RUN  ###########################3
# for train_tomos in training_schedule:
#     concat_train_ids = sorted([s.replace('tomo', '') for s in train_tomos])
#     concat_train_ids = '-'.join(concat_train_ids)

#     paths_trainData, paths_trainTarget = setup.get_paths([t for t in tomo_ids if t[0:6] in train_tomos])

#     if len(train_tomos)==1:
#         tb_logdir = os.path.join(PARENT_PATH, 'models_scratchpad/logs/LowBaselineModel/train%s' %concat_train_ids)
#         model_name = '3.00_lowBaseline'
#     elif len(train_tomos)==3:
#         tb_logdir = os.path.join(PARENT_PATH, 'models_scratchpad/logs/BaselineModel/train%s' %concat_train_ids)
#         model_name = '3.00_Baseline'

#     trainer = make_trainer(dim_in=84, batch_size=6, lr=1e-4, epochs=500, tb_logdir=tb_logdir, model_name=model_name,
#                        reconstruction_trainer=False, pretrained_model=None, test_run=True)

#     # print('Fitting only to ', paths_trainData[0:1])
#     trainer.launch(paths_trainData, paths_trainTarget, paths_valData, paths_valTarget)
