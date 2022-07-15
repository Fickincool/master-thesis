from tomoSegmentPipeline.utils.common import read_array, write_array
from tomoSegmentPipeline.utils import setup

from cryoS2Sdrop.dataloader import singleCET_dataset
from cryoS2Sdrop.model import Denoising_UNet
from cryoS2Sdrop.losses import self2self_L2Loss
from cryoS2Sdrop.trainer import denoisingTrainer
from cryoS2Sdrop.predict import load_model, predict_full_tomogram
import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob

from torchsummary import summary
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

PARENT_PATH = setup.PARENT_PATH

version = "version_34"
logdir = os.path.join(PARENT_PATH, "data/S2SDenoising/tryout_model_logs/%s/" % version)
# logdir = os.path.join(PARENT_PATH, 'data/S2SDenoising/model_logs/%s/' %version)

model, hparams = load_model(logdir, DataParallel=True)

cet_path = hparams["cet_path"]
p = float(hparams["p"])  # dropout (zeroing) probability
subtomo_length = int(hparams["subtomo_length"])
n_features = int(hparams["n_features"])

my_dataset = singleCET_dataset(cet_path, subtomo_length, p=p)
len(my_dataset)

#################### one averaged prediction from N model predictions on a single bernoulli sample

subtomo, target, mask = my_dataset[10]


def aux_forward(model):
    with torch.no_grad():
        subtomo, _, _ = my_dataset[10]
        return model(subtomo)


denoised_subtomo = (
    torch.stack([aux_forward(model) for i in range(50)])
    .mean((0, 1))
    .squeeze()
    .cpu()
    .detach()
    .numpy()
)

plt.figure(figsize=(12, 8))
plt.hist(denoised_subtomo.flatten(), alpha=0.5, label="Denoised subtomo")
plt.hist(subtomo.numpy().flatten(), alpha=0.5, label="Input subtomo")
plt.legend()
outfile = os.path.join(logdir, "subtomo_histograms.png")
plt.savefig(outfile)

original_subtomo = (target + subtomo)[0].squeeze().detach().numpy()

aux_idx = subtomo_length // 2

fig, (ax0, ax1) = plt.subplots(2, 3, figsize=(20, 15))
list(map(lambda axi: axi.set_axis_off(), np.array([ax0, ax1]).ravel()))

ax0[1].set_title("Original")
ax0[0].imshow(original_subtomo[aux_idx])
ax0[1].imshow(original_subtomo[:, aux_idx, :])
ax0[2].imshow(original_subtomo[:, :, aux_idx])

ax1[1].set_title("Denoised")
ax1[0].imshow(denoised_subtomo[aux_idx])
ax1[1].imshow(denoised_subtomo[:, aux_idx, :])
ax1[2].imshow(denoised_subtomo[:, :, aux_idx])

plt.tight_layout()
outfile = os.path.join(logdir, "subtomo original_vs_denoised.png")
plt.savefig(outfile)


##########################################################################################

################# Full tomogram ############################################################

torch.cuda.empty_cache()

batch_size = 10
denoised_tomo = []

print("Predicting full tomogram...")
# this is taking two means: first per bernoulli batches, and then again for each time the model was run
for i in tqdm(range(5)):
    _denoised_tomo = predict_full_tomogram(my_dataset, model, batch_size)
    denoised_tomo.append(_denoised_tomo)

# take mean over all bernoulli batches
denoised_tomo = torch.stack(denoised_tomo).mean(0)
# denoised_tomo = my_dataset.clip(denoised_tomo)
print("Done!")

plt.figure(figsize=(12, 8))
plt.hist(denoised_tomo.numpy().flatten(), alpha=0.5, label="Denoised")
plt.hist(my_dataset.data.numpy().flatten(), alpha=0.5, label="Original")
plt.legend()
outfile = os.path.join(logdir, "Full_tomo_histograms.png")
plt.savefig(outfile)

zidx, yidx, xidx = np.array(my_dataset.data.shape) // 2

fig, (ax0, ax1) = plt.subplots(2, 3, figsize=(25, 15))
list(map(lambda axi: axi.set_axis_off(), np.array([ax0, ax1]).ravel()))

ax0[1].set_title("Original: YX, ZX, ZY")
ax0[0].imshow(my_dataset.data[zidx])
ax0[1].imshow(my_dataset.data[:, yidx, :])
ax0[2].imshow(my_dataset.data[:, :, xidx])

ax1[1].set_title("Denoised: YX, ZX, ZY")
ax1[0].imshow(denoised_tomo[zidx])
ax1[1].imshow(denoised_tomo[:, yidx, :])
ax1[2].imshow(denoised_tomo[:, :, xidx])

plt.tight_layout()
outfile = os.path.join(logdir, "original_vs_denoised.png")
plt.savefig(outfile, dpi=200)
