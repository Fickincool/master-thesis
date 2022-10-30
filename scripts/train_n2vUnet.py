# We import all our dependencies.
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt
from tomoSegmentPipeline.utils import setup
from tomoSegmentPipeline.utils.common import read_array, write_array
from cryoS2Sdrop.deconvolution import tom_deconv_tomo
from cryoS2Sdrop.misc import parse_cet_paths

import urllib
import os
import zipfile
import json
import sys

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

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

##################################### Generate patches ####################################################

# We create our DataGenerator-object.
# It will help us load data and extract patches for training and validation.
datagen = N2V_DataGenerator()

# We will load all the '.tif' files from the 'data' directory. In our case it is only one.
# The function will return a list of images (numpy arrays).
# In the 'dims' parameter we specify the order of dimensions in the image files we are reading.
imgs = datagen.load_imgs([cet_path], dims="ZYX")

# We don't implement clipping nor standarization since N2V applies some preprocessing (normalization)
# if clip_values:
#     imgs[0][0, ..., 0] = clip(imgs[0][0, ..., 0])
# imgs[0][0, ..., 0] = standardize(imgs[0][0, ..., 0])

if use_deconv_data:
    if len(imgs) > 1:
        raise ValueError(
            "Deconvolution not implemented for more than 1 training tomogram."
        )
    deconv_kwargs["vol"] = imgs[0][0, ..., 0]
    imgs[0][0, ..., 0] = tom_deconv_tomo(**deconv_kwargs)

else:
    pass

# Let's look at the shape of the image
print("Image shape: ", imgs[0].shape)
# The function automatically added two extra dimension to the images:
# One at the front is used to hold a potential stack of images such as a movie.
# One at the end could hold color channels such as RGB.

# Here we extract patches for training and validation.
patch_shape = (96, 96, 96)
patches = datagen.generate_patches_from_list(imgs[:1], shape=patch_shape, shuffle=True)

# Patches are created so they do not overlap.
# (Note: this is not the case if you specify a number of patches. See the docstring for details!)
# Non-overlapping patches enable us to split them into a training and validation set.
n_training = int(0.9 * len(patches))
X = patches[:n_training]
X_val = patches[n_training:]

############################## Network configuration ##########################################

# You can increase "train_steps_per_epoch" to get even better results at the price of longer computation.
config = N2VConfig(
    X,
    unet_kern_size=3,
    train_steps_per_epoch=int(X.shape[0] / 16),
    train_epochs=epochs,
    train_loss="mse",
    batch_norm=True,
    train_batch_size=4,
    n2v_perc_pix=0.198,
    n2v_patch_shape=patch_shape,
    n2v_manipulator="uniform_withCP",
    n2v_neighborhood_radius=5,
)


# a name used to identify the model
if use_deconv_data:
    model_name = "deconv"
    # the base directory in which our model will live
    basedir = "/home/ubuntu/Thesis/data/S2SDenoising/n2v_model_logs/%s/" % name

else:
    model_name = "normal"
    # the base directory in which our model will live
    basedir = "/home/ubuntu/Thesis/data/S2SDenoising/n2v_model_logs/%s/" % name
# We are now creating our network model.
model = N2V(config=config, name=model_name, basedir=basedir)

############################## Training #############################

history = model.train(X, X_val)
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16, 5))
plot_history(history, ["loss", "val_loss"])
outfile = os.path.join(basedir, "%s/losses.png" % model_name)
plt.savefig(outfile, dpi=200)
