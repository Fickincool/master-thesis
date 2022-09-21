import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from time import sleep
from tomoSegmentPipeline.utils.common import write_array
from tomoSegmentPipeline.utils import setup
from cryoS2Sdrop.dataloader import (
    singleCET_dataset,
    singleCET_FourierDataset,
    singleCET_ProjectedDataset,
)
from cryoS2Sdrop.trainer import (
    denoisingTrainer,
    aggregate_bernoulliSamples,
    aggregate_bernoulliSamples2,
    collate_for_oneBernoulliSample,
)
from cryoS2Sdrop.dataloader import randomRotation3D, randomRotation3D_fourierSamples
from cryoS2Sdrop.losses import self2self_L2Loss, self2selfLoss, self2selfLoss_noMask
from cryoS2Sdrop.model import Denoising_3DUNet, Denoising_3DUNet_v2
from cryoS2Sdrop.predict import load_model, predict_full_tomogram
from pytorch_msssim import ssim
from torchmetrics.functional import peak_signal_noise_ratio
import sys
import json
import pathlib

PARENT_PATH = setup.PARENT_PATH

###################### Parse arguments ###################

args = json.loads(sys.argv[1])
exp_name = sys.argv[2]

p = args["p"]  # bernoulli masking probability
n_bernoulli_samples = args["n_bernoulli_samples"]
total_samples = args["total_samples"]
alpha = args["alpha"]
volumetric_scale_factor = args["volumetric_scale_factor"]
Vmask_probability = args["Vmask_probability"]
Vmask_pct = args["Vmask_pct"]

subtomo_length = args["subtomo_length"]
n_features = args["n_features"]
batch_size = args["batch_size"]
epochs = args["epochs"]
lr = args["lr"]
num_gpus = args["num_gpus"]
predict_simRecon = args["predict_simRecon"]
use_deconv_as_target = args["use_deconv_as_target"]

bernoulliMask_prob = args["bernoulliMask_prob"]
input_as_target = args["input_as_target"]

tomo_name = args["tomo_name"]

deconv_kwargs = args["deconv_kwargs"]

if tomo_name.startswith("tomoPhantom"):

    cet_path = os.path.join(
        PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s.mrc" % tomo_name
    )
    model_name = tomo_name.split("_")[1]

    gt_cet_path = os.path.join(
        PARENT_PATH, "data/S2SDenoising/dummy_tomograms/tomoPhantom_%s.mrc" % model_name
    )

elif tomo_name.startswith("tomo"):
    cet_path = os.path.join(
        PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s.mrc" % tomo_name
    )
    _name = tomo_name.split("_")[0]
    gt_cet_path = os.path.join(
        PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s_cryoCAREDummy.mrc" % _name
    )

elif tomo_name.startswith("shrec2021"):
    cet_path = os.path.join(
        PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s.mrc" % tomo_name
    )
    _name = tomo_name.split("_")[1]
    gt_cet_path = os.path.join(
        PARENT_PATH,
        "data/S2SDenoising/dummy_tomograms/shrec2021_%s_gtDummy.mrc" % _name,
    )
##################################### Model and dataloader ####################################################

tensorboard_logdir = os.path.join(
    PARENT_PATH, "data/S2SDenoising/model_logs/%s/%s/" % (tomo_name, exp_name)
)
pathlib.Path(tensorboard_logdir).mkdir(parents=True, exist_ok=True)

comment = args["comment"]

n_shift = 10

if args["dataset"] == "singleCET_dataset":
    my_dataset = singleCET_dataset(
        cet_path,
        subtomo_length=subtomo_length,
        p=p,
        n_bernoulli_samples=n_bernoulli_samples,
        volumetric_scale_factor=volumetric_scale_factor,
        Vmask_probability=Vmask_probability,
        Vmask_pct=Vmask_pct,
        transform=None,
        n_shift=n_shift,
        gt_tomo_path=gt_cet_path,
        **deconv_kwargs
    )

    collate_fn = aggregate_bernoulliSamples
    loss_fn = self2selfLoss(alpha=alpha)
    model = Denoising_3DUNet(loss_fn, lr, n_features, p, n_bernoulli_samples)
    model_name = "s2sDenoise3D"
    transform = randomRotation3D(0.5)
    my_dataset.transform = transform

elif args["dataset"] == "singleCET_FourierDataset":
    my_dataset = singleCET_FourierDataset(
        cet_path,
        subtomo_length=subtomo_length,
        p=p,
        n_bernoulli_samples=n_bernoulli_samples,
        total_samples=total_samples,
        volumetric_scale_factor=volumetric_scale_factor,
        Vmask_probability=Vmask_probability,
        Vmask_pct=Vmask_pct,
        transform=None,
        n_shift=n_shift,
        gt_tomo_path=gt_cet_path,
        bernoulliMask_prob=bernoulliMask_prob,
        input_as_target=input_as_target,
        **deconv_kwargs
    )

    collate_fn = aggregate_bernoulliSamples2
    loss_fn = self2selfLoss_noMask(alpha=alpha)
    model = Denoising_3DUNet_v2(loss_fn, lr, n_features, p, n_bernoulli_samples)
    model_name = "s2sDenoise3D_fourier"
    transform = randomRotation3D_fourierSamples(0.5)

elif args["dataset"] == "singleCET_ProjectedDataset":
    my_dataset = singleCET_ProjectedDataset(
        cet_path,
        subtomo_length=subtomo_length,
        transform=None,
        n_shift=n_shift,
        gt_tomo_path=gt_cet_path,
        predict_simRecon=predict_simRecon,
        use_deconv_as_target=use_deconv_as_target,
        **deconv_kwargs
    )

    collate_fn = collate_for_oneBernoulliSample
    # override bernoulli samples in this case. We always have only one.
    n_bernoulli_samples = 1
    loss_fn = self2selfLoss_noMask(alpha=alpha)
    model = Denoising_3DUNet_v2(loss_fn, lr, n_features, p, n_bernoulli_samples)
    model_name = "s2sDenoise3D_simulatedN2N"
    transform = randomRotation3D_fourierSamples(0.5)


##################################### Training ####################################################

s2s_trainer = denoisingTrainer(
    model, my_dataset, tensorboard_logdir, model_name=model_name
)

s2s_trainer.train(
    collate_fn, batch_size, epochs, num_gpus, transform=transform, comment=comment
)

version = "version_%i" % s2s_trainer.model.logger.version
logdir = os.path.join(tensorboard_logdir, "%s/" % version)

with open(os.path.join(logdir, "experiment_args.json"), "w") as f:
    json.dump(args, f)
