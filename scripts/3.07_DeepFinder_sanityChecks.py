#!/usr/bin/env python
# coding: utf-8

# In[11]:


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
import torch
import matplotlib.pyplot as plt

PARENT_PATH = setup.PARENT_PATH

train_tomos = ["tomo02", "tomo03", "tomo17"]
concat_train_ids = sorted([s.replace("tomo", "") for s in train_tomos])
concat_train_ids = "-".join(concat_train_ids)

val_tomos = ["tomo32", "tomo10"]
concat_val_ids = sorted([s.replace("tomo", "") for s in val_tomos])
concat_val_ids = "-".join(concat_val_ids)

test_tomos = ["tomo38", "tomo04"]
concat_test_ids = sorted([s.replace("tomo", "") for s in test_tomos])
concat_test_ids = "-".join(concat_test_ids)


# # Overfit one patch
#
# Use the dummyDataset, which yields only one patch from `tomo02` with dimensions `dim_in^3`

# In[6]:


# Ncl = 2
# dim_in = 52
# lr = 1e-4
# weight_decay = 0
# Lrnd = 0
# tensorboard_logdir = os.path.join(PARENT_PATH, 'model_sanityChecks/logs/overfit_to_patch/')
# model_name='3.07_testRun'
# augment_data = False
# batch_size = 1
# epochs = 500
# pretrained_model = None


#  ## Multiple GPUs

# # In[19]:


# tensorboard_logdir = os.path.join(PARENT_PATH, 'model_sanityChecks/logs/overfit_to_patch_multiGPU_ddp/')

# trainer_overfit1 = Train(Ncl, dim_in, lr, weight_decay, Lrnd, tensorboard_logdir, model_name, augment_data,
#                          batch_size, epochs, pretrained_model)

# trainer_overfit1.launch(train_tomos, val_tomos, input_type='cryoCARE', num_gpus=3, accelerator='ddp',
#                         num_workers=1, dataset=dl.tomoSegment_dummyDataset)


# # Train using only one patch

# In[34]:


Ncl = 2
dim_in = 52
lr = 1e-4
weight_decay = 0
Lrnd = 0
tensorboard_logdir = os.path.join(
    PARENT_PATH, "model_sanityChecks/logs/one_patch_training_ddp/"
)
model_name = "3.07_onePatch"
augment_data = False
batch_size = 32
epochs = 500
pretrained_model = None


# In[ ]:


trainer_1patch = Train(
    Ncl,
    dim_in,
    lr,
    weight_decay,
    Lrnd,
    tensorboard_logdir,
    model_name,
    augment_data,
    batch_size,
    epochs,
    pretrained_model,
)

trainer_1patch.launch(
    train_tomos,
    val_tomos,
    input_type="cryoCARE",
    num_gpus=3,
    accelerator="ddp",
    num_workers=1,
    dataset=dl.tomoSegment_dataset,
)
