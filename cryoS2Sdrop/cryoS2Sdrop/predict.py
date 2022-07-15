from cryoS2Sdrop.model import Denoising_UNet
from cryoS2Sdrop.trainer import aggregate_bernoulliSamples
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from glob import glob


def aux_forward(model, subtomo):
    with torch.no_grad():
        return model(subtomo)


def load_model(logdir, DataParallel=False):
    "Returns loaded model from checkpoint and hyperparameters"

    with open(glob(logdir + "hparams.yaml")[0]) as f:
        hparams = yaml.load(f, Loader=yaml.BaseLoader)

    ckpt_file = glob(logdir + "checkpoints/*.ckpt")
    assert len(glob(logdir + "checkpoints/*.ckpt")) == 1
    ckpt_file = ckpt_file[0]

    model = Denoising_UNet.load_from_checkpoint(ckpt_file).cuda()
    if DataParallel:
        model = torch.nn.DataParallel(model)

    return model, hparams


def predict_full_tomogram(singleCET_dataset, model, batch_size):

    tomo_shape = singleCET_dataset.tomo_shape
    subtomo_length = singleCET_dataset.subtomo_length

    dloader = DataLoader(
        singleCET_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    denoised_tomo = torch.zeros(tomo_shape)
    count_tensor = torch.zeros(tomo_shape)  # for averaging overlapping patches

    for idx, batch in enumerate(dloader):

        subtomo, _, _ = batch  # shape: [B, M, C:=1, S, S, S]

        # one prediction per subtomogram (batch member) averaging over bernoulli samples
        denoised_subtomo = (
            torch.stack([aux_forward(model, s) for s in subtomo])
            .mean(1)
            .squeeze(1)
            .cpu()
        )

        grid_min, grid_max = idx * batch_size, (idx + 1) * batch_size
        grid_max = min(grid_max, len(singleCET_dataset))
        for batch_idx, grid_idx in enumerate(range(grid_min, grid_max)):

            z0, y0, x0 = singleCET_dataset.grid[grid_idx]
            zmin, zmax = z0 - subtomo_length // 2, z0 + subtomo_length // 2
            ymin, ymax = y0 - subtomo_length // 2, y0 + subtomo_length // 2
            xmin, xmax = x0 - subtomo_length // 2, x0 + subtomo_length // 2

            count_tensor[zmin:zmax, ymin:ymax, xmin:xmax] += 1
            denoised_tomo[zmin:zmax, ymin:ymax, xmin:xmax] += denoised_subtomo[
                batch_idx
            ]

    # Get average predictions for overlapping patches
    denoised_tomo = denoised_tomo / count_tensor
    del count_tensor

    return denoised_tomo
