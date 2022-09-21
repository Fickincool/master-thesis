from tomoSegmentPipeline.utils.common import read_array, write_array
from tomoSegmentPipeline.utils import setup
from cryoS2Sdrop.predict import load_model

import seaborn as sns

sns.set_style("dark")

import matplotlib as mpl

mpl.rcParams["figure.titlesize"] = 16
mpl.rc("axes", labelsize=12)
mpl.rc("image", cmap="viridis")

from pytorch_msssim import ssim
from torchmetrics.functional import peak_signal_noise_ratio, mean_squared_error
import torch
import string
from IPython.display import display

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import yaml

PARENT_PATH = setup.PARENT_PATH


def standardize(X: torch.tensor):
    mean = X.mean()
    std = X.std()
    new_X = (X - mean) / std

    return new_X


def clip(X, low=0.005, high=0.995):
    # works with tensors =)
    return np.clip(X, np.quantile(X, low), np.quantile(X, high))


def scale(X):
    scaled = (X - X.min()) / (X.max() - X.min() + 1e-5)
    return scaled


def get_metrics(tomo_path, gt_tomo_path, use_deconv_data, clip_values):

    n2v_psnr, n2v_ssim_idx = None, None

    if use_deconv_data == "true":
        use_deconv_data = True
    elif use_deconv_data == "false":
        use_deconv_data = False

    if (tomo_path is not None) and (gt_tomo_path is not None):
        try:
            gt_data = read_array(gt_tomo_path)
            gt_data = torch.tensor(gt_data).unsqueeze(0).unsqueeze(0)
            if clip_values:
                gt_data = clip(gt_data)
            gt_data = standardize(gt_data)

            name = tomo_path.split("/")[-1].replace(".mrc", "")

            if use_deconv_data:
                _type = "deconv"
            else:
                _type = "normal"

            n2v_pred_path = os.path.join(
                PARENT_PATH,
                "data/S2SDenoising/n2v_model_logs/%s/%s/%s_n2vDenoised.mrc"
                % (name, _type, name),
            )
            # print(n2v_pred_path)

            n2v_data = read_array(n2v_pred_path)
            n2v_data = torch.tensor(n2v_data).unsqueeze(0).unsqueeze(0)
            # here we clip and standardize n2v data because preprocessing is included in the original implementation
            # and it is slightly different to what we do.
            # also, we use clipped values for training our network, so a valid comparison needs to include clipped values
            # of the other denoising schemes.
            if clip_values:
                n2v_data = clip(n2v_data)
            n2v_data = standardize(n2v_data)
            n2v_data = scale(n2v_data)
            gt_data = scale(gt_data)

            n2v_psnr = float(peak_signal_noise_ratio(n2v_data, gt_data, data_range=1))
            n2v_ssim_idx = float(ssim(n2v_data, gt_data, data_range=1))

        except OSError:
            pass

    return n2v_psnr, n2v_ssim_idx


def parse_noise_level(x):
    if x is not None:
        if "tomoPhantom" in x:
            if "Perlin" not in x:
                x = x.split("_")[-1].replace(".mrc", "")
                label_to_var = {"VL": 0.2, "L": 0.5, "M": 1, "H": 5}
                for k in label_to_var.keys():
                    if k in x:
                        return "Gauss(%.1f) + Poisson" % label_to_var[k]
            elif "Perlin" in x:
                x = x.split("noisy")[-1].replace(".mrc", "")
                label_to_var = {
                    "VL_Perlin": 0.2,
                    "L_Perlin": 0.5,
                    "M_Perlin": 1,
                    "H_Perlin": 5,
                }
                for k in label_to_var.keys():
                    if k in x:
                        return "Gauss(%.1f) + Poisson + Perlin" % label_to_var[k]
        elif "shrec" in x:
            return "other"
    else:
        pass


def parse_tomo_name(name):
    if name.startswith("tomoPhantom_model14"):
        name = "cell"
    if name.startswith("tomoPhantom_model16"):
        name = "spaceship"
    if name.startswith("tomoPhantom_model8"):
        name = "blobs"

    return name


def parse_pred_tomo_path(data_log):

    version = data_log.version.values
    tomo_name = data_log.tomo_path.map(lambda x: x.split("/")[-1].replace(".mrc", ""))
    exp_name = data_log.model.unique()[0]
    exp_name = len(tomo_name) * [exp_name]

    data_log["tomo_name"] = [parse_tomo_name(t) for t in tomo_name]
    logdir = [
        "data/S2SDenoising/model_logs/%s/%s/" % (t, e)
        for t, e in zip(tomo_name, exp_name)
    ]
    logdir = [os.path.join(PARENT_PATH, l) for l in logdir]

    pred_tomo_path = [
        l + "%s/%s_s2sDenoised.mrc" % (v, t)
        for l, v, t in zip(logdir, version, tomo_name)
    ]
    data_log["pred_path"] = pred_tomo_path

    return data_log


def logdir_to_dataframe(logdir, clip_values, ignore_deconv=True):
    data_log = []
    keys = [
        "Version_comment",
        "transform",
        "full_tomo_psnr",
        "full_tomo_ssim",
        "baseline_psnr",
        "baseline_ssim",
        "tomo_path",
        "gt_tomo_path",
        "use_deconv_as_target",
        "predict_simRecon",
        "use_deconv_data",
        "p",
    ]

    all_logs = glob(logdir + "*/*.yaml")

    for yaml_logdir in all_logs:
        model = yaml_logdir.split("/")[-3]
        version = yaml_logdir.split("/")[-2]
        with open(yaml_logdir) as f:
            hparams = yaml.load(f, Loader=yaml.BaseLoader)

        if "dataset" in hparams.keys():
            dataset = hparams["dataset"]

        else:
            # yaml is stupid
            with open(yaml_logdir) as f:
                s = f.readlines()
                try:
                    dataloader = [x for x in s if "Dataloader" in x][0]
                    dataset = (
                        dataloader.split(".")[-1].replace("\n", "").replace("'", "")
                    )
                except:
                    dataset = "Unknown"

        row_vals = [model, version, dataset, hparams["loss_fn"]["alpha"]]
        for k in keys:
            try:
                row_vals += [hparams[k]]
            except KeyError:
                row_vals += [None]
        data_log.append(row_vals)

    data_log = pd.DataFrame(
        data_log, columns=["model", "version", "dataset", "TV_alpha"] + keys
    )
    data_log = data_log.sort_values(["model", "version"])

    if ignore_deconv:
        baseline_metrics = data_log[
            ["tomo_path", "gt_tomo_path", "use_deconv_data"]
        ].apply(lambda x: get_metrics(x[0], x[1], "false", clip_values), axis=1)
    else:
        baseline_metrics = data_log[
            ["tomo_path", "gt_tomo_path", "use_deconv_data"]
        ].apply(lambda x: get_metrics(x[0], x[1], x[2], clip_values), axis=1)
    data_log["n2v_psnr"], data_log["n2v_ssim"] = zip(*baseline_metrics)
    data_log[["full_tomo_psnr", "full_tomo_ssim"]] = data_log[
        ["full_tomo_psnr", "full_tomo_ssim"]
    ].astype(float)
    data_log[["baseline_psnr", "baseline_ssim"]] = data_log[
        ["baseline_psnr", "baseline_ssim"]
    ].astype(float)

    data_log["baseline_psnr_best"] = data_log.baseline_psnr.max()
    data_log["baseline_ssim_best"] = data_log.baseline_ssim.max()

    tomo_path = (
        data_log.tomo_path
    )  # .map(lambda x: x.split('/')[-1] if x is not None else x)
    gt_tomo_path = (
        data_log.gt_tomo_path
    )  # .map(lambda x: x.split('/')[-1] if x is not None else x)

    # data_log.drop(['tomo_path', 'gt_tomo_path'], axis=1, inplace=True)
    data_log["tomo_path"], data_log["gt_tomo_path"] = [tomo_path, gt_tomo_path]

    _vals = 100 * data_log[["full_tomo_ssim", "n2v_ssim"]].apply(
        lambda x: (x - data_log["baseline_ssim_best"]) / data_log["baseline_ssim_best"]
    )
    data_log["ssim_vs_baseline"], data_log["n2v_ssim_vs_baseline"] = zip(*_vals.values)

    _vals = 100 * data_log[["full_tomo_psnr", "n2v_psnr"]].apply(
        lambda x: (x - data_log["baseline_psnr_best"]) / data_log["baseline_psnr_best"]
    )
    data_log["psnr_vs_baseline"], data_log["n2v_psnr_vs_baseline"] = zip(*_vals.values)

    data_log["noise_level"] = data_log.tomo_path.map(lambda x: parse_noise_level(x))

    data_log = parse_pred_tomo_path(data_log)

    return data_log


def get_best_version(data_log, metric):
    if metric == "psnr":
        _best = data_log[data_log.full_tomo_psnr == data_log.full_tomo_psnr.max()]
        _worst = data_log[data_log.full_tomo_psnr == data_log.full_tomo_psnr.min()]
    if metric == "ssim":
        _best = data_log[data_log.full_tomo_ssim == data_log.full_tomo_ssim.max()]
        _worst = data_log[data_log.full_tomo_ssim == data_log.full_tomo_ssim.min()]

    pred_tomo_path = _best.pred_path.values[0]
    gt_tomo_path = _best.gt_tomo_path.values[0]

    pred_tomo_path2 = _worst.pred_path.values[0]

    print("Best version: ", pred_tomo_path)
    print("GT file: ", gt_tomo_path)
    print("Worst version: ", pred_tomo_path2)
    print("")
    print("Best %s value: " % metric, _best["full_tomo_%s" % metric])
    print("Worst %s value: " % metric, _worst["full_tomo_%s" % metric])

    best = read_array(pred_tomo_path)
    gt = read_array(gt_tomo_path)
    worst = read_array(pred_tomo_path2)

    return best, gt, worst
