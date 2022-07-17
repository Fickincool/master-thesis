#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tomoSegmentPipeline.showcaseResults import (
    predict_fullTomogram,
    load_model,
    load_tomoData,
    Tversky_index,
    fullTomogram_modelComparison,
    make_comparison_plot,
    write_comparison_gif,
    save_classPred,
)

from tomoSegmentPipeline.losses import Tversky_loss
from tomoSegmentPipeline.utils.common import read_array
import tomoSegmentPipeline.dataloader as dl
from tomoSegmentPipeline.utils import setup
from tomoSegmentPipeline.dataloader import to_categorical, transpose_to_channels_first
from tomoSegmentPipeline.trainWrapper import make_trainer

PARENT_PATH = setup.PARENT_PATH

import numpy as np
import matplotlib.pyplot as plt
import random
import mrcfile
import pandas as pd
import torch
import os
from glob import glob
import random

train_tomos = ["tomo02", "tomo03", "tomo17"]
concat_train_ids = sorted([s.replace("tomo", "") for s in train_tomos])
concat_train_ids = "-".join(concat_train_ids)

val_tomos = ["tomo32", "tomo10"]
concat_val_ids = sorted([s.replace("tomo", "") for s in val_tomos])
concat_val_ids = "-".join(concat_val_ids)

test_tomos = ["tomo38", "tomo04"]
concat_test_ids = sorted([s.replace("tomo", "") for s in test_tomos])
concat_test_ids = "-".join(concat_test_ids)


paths_trainData, paths_trainTarget = setup.get_paths(train_tomos, "cryoCARE")
paths_valData, paths_valTarget = setup.get_paths(val_tomos, "cryoCARE")
paths_testData, paths_testTarget = setup.get_paths(test_tomos, "cryoCARE")


# In[5]:


from tomoSegmentPipeline import dataloader as dl
from tomoSegmentPipeline.utils import setup
from tomoSegmentPipeline.dataloader import (
    to_categorical,
    transpose_to_channels_first,
    tomoSegment_dummyDataset,
    tomoSegment_dataset,
)
from tomoSegmentPipeline.training import Train
import os

PARENT_PATH = setup.PARENT_PATH

import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


val_tomos = ["tomo32", "tomo10"]
concat_val_ids = sorted([s.replace("tomo", "") for s in val_tomos])
concat_val_ids = "-".join(concat_val_ids)

test_tomos = ["tomo38", "tomo04"]
concat_test_ids = sorted([s.replace("tomo", "") for s in test_tomos])
concat_test_ids = "-".join(concat_test_ids)

train_tomos = ["tomo02"]

concat_train_ids = sorted([s.replace("tomo", "") for s in train_tomos])
concat_train_ids = "-".join(concat_train_ids)

chkpnt = None

tb_logdir = os.path.join(
    PARENT_PATH,
    "models_scratchpad2/earlyStop_tryout/logs/LowBaselineModel/train%s"
    % concat_train_ids,
)
model_name = "3.09_lowBaseline"
epochs = 100


Ncl = 2
dim_in = 56
lr = 3e-4
weight_decay = 0
Lrnd = 0
augment_data = False
batch_size = 56
nPatches_training = 1
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
    monitor="hp/val_loss_epoch",
    min_delta=1e-4,
    patience=6,
    verbose=True,
    mode="min",
    # check_on_train_epoch_end=False
)

trainer.launch(
    train_tomos,
    val_tomos,
    input_type="cryoCARE",
    num_gpus=3,
    accelerator="ddp",
    num_workers=1,
    resume_from_checkpoint=chkpnt,
    nPatches_training=nPatches_training,
    train_callbacks=[early_stop_callback],
)


# In[ ]:
