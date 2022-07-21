from tomoSegmentPipeline import dataloader as dl
from tomoSegmentPipeline.utils import setup
from tomoSegmentPipeline.dataloader import (
    to_categorical,
    transpose_to_channels_first,
    tomoSegment_dummyDataset,
    tomoSegment_dataset,
)
from tomoSegmentPipeline.training import Train
from tomoSegmentPipeline.showcaseResults import (
    predict_fullTomogram,
    load_model,
    load_tomoData,
    Tversky_index,
    Tversky_loss,
    fullTomogram_modelComparison,
    make_comparison_plot,
    write_comparison_gif,
    save_classPred,
)

from tomoSegmentPipeline.model import DeepFinder_model

import pytorch_lightning as pl

import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader


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


logs_path = PARENT_PATH + "data/model_logs/models_1/logs/BaselineModel/"
logs_path = Path(logs_path)

model_info = []

logdir_path = "/home/haicu/jeronimo.carvajal/Thesis/data/model_logs/models_1/logs/BaselineModel/cryoCARE/train02-03-17/version_2/"

model_file = glob(os.path.join(logdir_path, "*.model"))

model_file = model_file[0]

model_file_split = model_file.split("/")

input_type = model_file_split[-4]

name, epochs, patch_size, lr, version = model_file_split[-1].split("_")
epochs = int(epochs.replace("ep", ""))
version = "v" + version.replace(".model", "")

events_path = glob(os.path.join(logdir_path, "events.*"))[0]
event_acc = EventAccumulator(events_path)
event_acc.Reload()

_, step_nums, values_valLoss = zip(*event_acc.Scalars("hp/val_loss_epoch"))
best_val_loss_epoch = np.min(values_valLoss)
best_val_loss_epoch_idx = np.argmin(values_valLoss)  # index starts count at 0

effective_epochs = len(values_valLoss)

_, _, values_dice = zip(*event_acc.Scalars("hp/val_dice_epoch"))
_, _, values_trainLoss = zip(*event_acc.Scalars("hp/train_loss_epoch"))

associated_val_class1_dice = float(values_dice[best_val_loss_epoch_idx])
associated_train_loss_epoch = float(values_trainLoss[best_val_loss_epoch_idx])

epochs_str = "%i out of %i" % (effective_epochs, 1000)

model_info.append(
    [
        name,
        model_file,
        input_type,
        epochs_str,
        patch_size,
        lr,
        version,
        best_val_loss_epoch,
        associated_val_class1_dice,
    ]
)

df_model = pd.DataFrame(
    model_info,
    columns=[
        "name",
        "model_file",
        "input_type",
        "epochs",
        "patch_size",
        "lr",
        "version",
        "best_val_loss_epoch",
        "associated_val_class1_dice",
    ],
)
df_model.head()

dim_in = int(patch_size.replace("in", ""))

paths_valData, paths_valTarget = setup.get_paths(val_tomos, input_type)

my_dataset = dl.tomoSegment_dataset(
    paths_valData, paths_valTarget, dim_in=dim_in, Ncl=3, Lrnd=0, augment_data=False
)
val_loader = DataLoader(
    my_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=1
)

loss_fn = Tversky_loss()
# model = DeepFinder_model(2, loss_fn, 1e-4, 0, None)

model = load_model(model_file, 2)

trainer = pl.Trainer(gpus=3, strategy="ddp")

aux_model_name = model_file.split("/")[-1]
ckpt_file = model_file.replace(aux_model_name, "checkpoints/")
ckpt_file = glob(ckpt_file + "*")[0]
ckpt_file = None

trainer.validate(model, dataloaders=val_loader, ckpt_path=ckpt_file)
