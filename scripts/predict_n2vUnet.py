# We import all our dependencies.
from n2v.models import N2VConfig, N2V
import numpy as np
from matplotlib import pyplot as plt
from tomoSegmentPipeline.utils import setup
from tomoSegmentPipeline.utils.common import read_array, write_array
from cryoS2Sdrop.deconvolution import tom_deconv_tomo
from cryoS2Sdrop.misc import parse_cet_paths

import os
import json
import sys


PARENT_PATH = setup.PARENT_PATH


def standardize(X):
    mean = X.mean()
    std = X.std()

    new_X = (X - mean) / std

    return new_X


def scale(X):
    scaled = (X - X.min()) / (X.max() - X.min() + 1e-8)
    return scaled


def clip(X, low=0.005, high=0.995):
    # works with tensors =)
    return np.clip(X, np.quantile(X, low), np.quantile(X, high))


def check_deconv_kwargs(deconv_kwargs):
    if bool(deconv_kwargs):
        deconv_args = [
            "angpix",
            "defocus",
            "snrfalloff",
            "deconvstrength",
            "highpassnyquist",
        ]
        for arg in deconv_args:
            if arg in deconv_kwargs.keys():
                continue
            else:
                raise KeyError('Missing required deconvolution argument: "%s"' % arg)
        use_deconv_data = True
        print("Using deconvolved data for training.")

    else:
        use_deconv_data = False

    return use_deconv_data


###################### Input data definition ###################

args = json.loads(sys.argv[1])
exp_name = sys.argv[2]

tomo_name = args["tomo_name"]
epochs = args["epochs"]

cet_path, _ = parse_cet_paths(PARENT_PATH, tomo_name)

deconv_kwargs = args["deconv_kwargs"]
use_deconv_data = check_deconv_kwargs(deconv_kwargs)

gt_cet_path = None
clip_values = True

name = tomo_name

# a name used to identify the model
if use_deconv_data:
    model_name = "deconv"
    # the base directory in which our model will live
    basedir = "/home/ubuntu/Thesis/data/S2SDenoising/n2v_model_logs/%s/" % name

else:
    model_name = "normal"
    # the base directory in which our model will live
    basedir = "/home/ubuntu/Thesis/data/S2SDenoising/n2v_model_logs/%s/" % name

############################# Prediction #########################

model = N2V(config=None, name=model_name, basedir=basedir)

img = read_array(cet_path)

if use_deconv_data:
    deconv_kwargs["vol"] = img
    img = tom_deconv_tomo(**deconv_kwargs)

else:
    pass

pred = model.predict(img, axes="ZYX", n_tiles=(4, 4, 4))

zidx, yidx, xidx = np.array(img.shape) // 2

fig, (ax0, ax1) = plt.subplots(2, 3, figsize=(25, 15))
list(map(lambda axi: axi.set_axis_off(), np.array([ax0, ax1]).ravel()))

ax0[1].set_title("Original: YX, ZX, ZY")
ax0[0].imshow(img[zidx])
ax0[1].imshow(img[:, yidx, :])
ax0[2].imshow(img[:, :, xidx])

ax1[1].set_title("Denoised: YX, ZX, ZY")
ax1[0].imshow(pred[zidx])
ax1[1].imshow(pred[:, yidx, :])
ax1[2].imshow(pred[:, :, xidx])

plt.tight_layout()
outfile = os.path.join(basedir, "%s/original_vs_N2Vdenoised.png" % model_name)
plt.savefig(outfile, dpi=200)

filename = cet_path.split("/")[-1].replace(".mrc", "_n2vDenoised")
filename = os.path.join(basedir, "%s/%s.mrc" % (model_name, filename))

write_array(pred, filename)
