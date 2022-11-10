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


def paths2dataDict(pathDict):
    dataDict = []
    
    # as of 20.12 we used a different clipping for shrec, also gt_data needs to be processed differently
    if 'shrec' in pathDict['raw_path']:
        for x in tqdm(pathDict.keys()):
            if pathDict[x] is not None and x!='gt_path':
                val = clip(read_array(pathDict[x]), low=0.0005, high=0.9995)
                val = scale(standardize(val))
            elif pathDict[x] is not None and x=='gt_path':
                val = -1*read_array(pathDict[x])
                val = val - val.min()
                val = clip(val, low=0.0005, high=0.9995)
                val = scale(standardize(val))
            else:
                val = None
            
            dataDict.append(val)
                    
    else:
        for x in tqdm(pathDict.keys()):
            if pathDict[x] is not None:
                val = clip(read_array(pathDict[x]), low=0.005, high=0.995)
                val = scale(standardize(val))
            else:
                val = None  
                
            dataDict.append(val)
        
    dataDict = dict(zip(pathDict.keys(), dataDict))    
    
    
    return dataDict

# tomoPhantom
def get_tomoPhantom_dataDict(model_no=8):
    raw_path = '/home/ubuntu/Thesis/data/S2SDenoising/dummy_tomograms/tomoPhantom_model%i_noisyGaussPoissVL_Perlin.mrc' %model_no
    deconv_path = None
    cryoCARE_path = None
    N2V_path = "/home/ubuntu/Thesis/data/S2SDenoising/n2v_model_logs/tomoPhantom_model%i_noisyGaussPoissVL_Perlin/normal/tomoPhantom_model%i_noisyGaussPoissVL_Perlin_n2vDenoised.mrc" %(model_no, model_no)
    S2Sd_path = '/home/ubuntu/Thesis/data/S2SDenoising/model_logs/tomoPhantom_model%i_noisyGaussPoissVL_Perlin/structuredNoise_comparison/version_1/tomoPhantom_model%i_noisyGaussPoissVL_Perlin_s2sDenoised.mrc' %(model_no, model_no)
    F2Fd_path = '/home/ubuntu/Thesis/data/S2SDenoising/model_logs/tomoPhantom_model%i_noisyGaussPoissVL_Perlin/structuredNoise_comparison/version_0/tomoPhantom_model%i_noisyGaussPoissVL_Perlin_s2sDenoised.mrc' %(model_no, model_no)
    isoNet_path = None
    gt_path = '/home/ubuntu/Thesis/data/S2SDenoising/dummy_tomograms/tomoPhantom_model%i.mrc' %model_no
    
    pathDict = {
        'raw_path':raw_path,
        'deconv_path':deconv_path,
        'cryoCARE_path':cryoCARE_path,
        'N2V_path':N2V_path,
        'S2Sd_path':S2Sd_path,
        'F2Fd_path':F2Fd_path,
        'isoNet_path':isoNet_path,
        'gt_path':gt_path,
    }
    
    return paths2dataDict(pathDict)

# SHREC 21
def get_shrec_dataDict(model_no=2):
    raw_path = '/home/ubuntu/Thesis/data/shrec2021/model_%i/reconstruction.mrc' %model_no
    deconv_path = '/home/ubuntu/Thesis/data/isoNet/SHREC_dataset/SHREC_tomos_deconv/model_%i.mrc' %model_no
    if model_no==2:
        cryoCARE_path = '/home/ubuntu/Thesis/data/shrec2021/model_%i/cryoCARE_reconstruction.mrc' %model_no
    else:
        cryoCARE_path = None
    N2V_path = '/home/ubuntu/Thesis/data/S2SDenoising/n2v_model_logs/shrec2021_model_%i/normal/reconstruction_n2vDenoised.mrc' %model_no    
    S2Sd_path = '/home/ubuntu/Thesis/data/S2SDenoising/model_logs/shrec2021_model_%i/samplingStrategy_comparison/version_1/reconstruction_s2sDenoised.mrc' %model_no
    F2Fd_path = '/home/ubuntu/Thesis/data/S2SDenoising/model_logs/shrec2021_model_%i/samplingStrategy_comparison/version_0/reconstruction_s2sDenoised.mrc' %model_no
    # if model_no in [2]:
    #     isoNet_path = '/home/ubuntu/Thesis/data/isoNet/single_image_SHREC_dataset/model_%i/corrected/model_%i_corrected.mrc' %(model_no, model_no)
    # else:
    print("WARNING: comparing to isoNet trained on many images.")
    isoNet_path = '/home/ubuntu/Thesis/data/isoNet/SHREC_dataset/SHREC_corrected/model_%i_corrected.mrc' %model_no
    gt_path = '/home/ubuntu/Thesis/data/shrec2021/model_%i/grandmodel.mrc' %model_no
    
    pathDict = {
        'raw_path':raw_path,
        'deconv_path':deconv_path,
        'cryoCARE_path':cryoCARE_path,
        'N2V_path':N2V_path,
        'S2Sd_path':S2Sd_path,
        'F2Fd_path':F2Fd_path,
        'isoNet_path':isoNet_path,
        'gt_path':gt_path,
    }
    
    return paths2dataDict(pathDict)

# spinach
def get_spinach_dataDict(tomo_no=2, keys=None):
    raw_path = '/home/ubuntu/Thesis/data/raw_cryo-ET/tomo%02i.mrc' %tomo_no
    deconv_path = '/home/ubuntu/Thesis/data/isoNet/RAW_dataset/RAW_allTomos_deconv/tomo%02i.mrc' %tomo_no
    cryoCARE_path = '/home/ubuntu/Thesis/data/nnUnet/nifti_files/tomo%02i_bin4_denoised_0000.nii.gz' %tomo_no
    N2V_path = '/home/ubuntu/Thesis/data/S2SDenoising/n2v_model_logs/tomo%02i/normal/tomo%02i_n2vDenoised.mrc' %(tomo_no, tomo_no)
    S2Sd_path = '/home/ubuntu/Thesis/data/S2SDenoising/model_logs/tomo%02i/raw_denoising/version_1/tomo%02i_s2sDenoised.mrc' %(tomo_no, tomo_no)
    F2Fd_path = '/home/ubuntu/Thesis/data/S2SDenoising/model_logs/tomo%02i/raw_denoising/version_0/tomo%02i_s2sDenoised.mrc' %(tomo_no, tomo_no)
    
    # if tomo_no==2:
    #     deconv_F2Fd_path = '/home/ubuntu/Thesis/data/S2SDenoising/model_logs/tomo%02i/deconv_denoising/version_0/tomo%02i_s2sDenoised.mrc' %(tomo_no, tomo_no)
    # else:
    #     deconv_F2Fd_path = None
    
    # if tomo_no in [2, 4]:
    # if False:
    #     isoNet_path = '/home/ubuntu/Thesis/data/isoNet/single_image_RAW_dataset/tomo%02i/corrected/tomo%02i_corrected.mrc' %(tomo_no, tomo_no)
    # else:
    print("WARNING: comparing to isoNet trained on many images.")
    isoNet_path = '/home/ubuntu/Thesis/data/isoNet/RAW_dataset/RAW_corrected/tomo%02i_corrected.mrc' %tomo_no
    gt_path = None

    pathDict = {
        'raw_path':raw_path,
        'deconv_path':deconv_path,
        'cryoCARE_path':cryoCARE_path,
        'N2V_path':N2V_path,
        'S2Sd_path':S2Sd_path,
        'F2Fd_path':F2Fd_path,
        # 'deconv_F2Fd_path':deconv_F2Fd_path, # the results of this are not better than F2Fd only
        'isoNet_path':isoNet_path,
        'gt_path':gt_path,
    }
    
    if keys is None:
        dataDict = paths2dataDict(pathDict)

    else:
        new_pathDict = [pathDict[k] for k in keys]
        new_pathDict = dict(zip(keys, new_pathDict))

        dataDict = paths2dataDict(new_pathDict)
    
    return dataDict


def plot_centralSlices(tomo_data, set_axis_off, names=None, use_global_minMax=False):
    shape = np.array(tomo_data.shape)
    idx_central_slices = shape//2
    ratios = shape/shape.max()
    
    fig, ax = plt.subplots(1, 3, figsize=(25, 10), gridspec_kw={'width_ratios': ratios})  
    if set_axis_off:
        list(map(lambda axi: axi.set_axis_off(), ax.ravel()))
        
    plt.tight_layout()
    
    if names is None:
        names = ['Central XY plane', 'Central ZX plane', 'Central ZY plane']
    
    for i in range(3):
        tomo_slice = np.take(tomo_data, idx_central_slices[i], axis=i)

        if use_global_minMax:
            ax[i].imshow(tomo_slice, vmin=tomo_data.min(), vmax=tomo_data.max())
        else:
            ax[i].imshow(tomo_slice)

        ax[i].set_title(names[i], fontsize=16)
    
    return fig, ax

def standardize(X: torch.tensor):
    mean = X.mean()
    std = X.std()
    new_X = (X - mean) / std

    return new_X

# for SHREC we used 0.0005 and 0.9995
# spinach and tomoPhantom 0.005, 0.995
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

def _tomo_name(tomo_path):

    divided_path = tomo_path.split("/")

    if 'shrec2021/model_' in tomo_path:
        shrec = divided_path[-3]
        model_num = divided_path[-2]
        tomo_name = shrec+'_'+model_num
    else:
        tomo_name = divided_path[-1].replace(".mrc", "")
    
    return tomo_name

def _parse_pred_path(logdir, version, tomo_name):
    if tomo_name.startswith('shrec2021_model_'):
        pred_path = pred_path = logdir + "%s/reconstruction_s2sDenoised.mrc" % (version)
    else:
        pred_path = logdir + "%s/%s_s2sDenoised.mrc" % (version, tomo_name)
        
    return pred_path

def parse_pred_tomo_path(data_log):

    version = data_log.version.values
    tomo_name = data_log.tomo_path.map(lambda x: _tomo_name(x))
    exp_name = data_log.model.unique()[0]
    exp_name = len(tomo_name) * [exp_name]

    data_log["tomo_name"] = [parse_tomo_name(t) for t in tomo_name]
    logdir = [
        "data/S2SDenoising/model_logs/%s/%s/" % (t, e)
        for t, e in zip(tomo_name, exp_name)
    ]
    logdir = [os.path.join(PARENT_PATH, l) for l in logdir]

    pred_tomo_path = [
        _parse_pred_path(l, v, t) for l, v, t in zip(logdir, version, tomo_name)
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
                val = hparams[k]
                if val=='null':
                    val = None
                row_vals += [val]
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

    try:
        data_log = parse_pred_tomo_path(data_log)
    except AttributeError:
        pass

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

def parse_tomoPaths(tomo_name):
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
        
    return cet_path, gt_cet_path
