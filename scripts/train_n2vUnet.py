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

import urllib
import os
import zipfile

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

PARENT_PATH = setup.PARENT_PATH

###################### Input data definition ###################

# cet_path = os.path.join(PARENT_PATH, 'data/raw_cryo-ET/tomo02.mrc')
# cet_path = os.path.join(PARENT_PATH, 'data/S2SDenoising/dummy_tomograms/tomo04_deconvDummy.mrc')
tomo_name = 'tomo02_cryoCAREDummy_noisyGaussPoiss'
cet_path = os.path.join(
    PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s.mrc" %tomo_name
)

gt_cet_path = None

name = tomo_name

##################################### Generate patches ####################################################

# We create our DataGenerator-object.
# It will help us load data and extract patches for training and validation.
datagen = N2V_DataGenerator()

# We will load all the '.tif' files from the 'data' directory. In our case it is only one.
# The function will return a list of images (numpy arrays).
# In the 'dims' parameter we specify the order of dimensions in the image files we are reading.
imgs = datagen.load_imgs([cet_path], dims='ZYX')

deconv_kwargs = {
    'angpix': 14,
    'defocus': 0,
    'snrfalloff': 1,
    'deconvstrength': 1,
    'highpassnyquist': 0.3
                 }

deconv_kwargs['vol'] = imgs[0][0,...,0]

if len(imgs)>1:
    raise ValueError("Deconvolution not implemented for more than 1 training tomogram.")

use_deconv_data = True

if use_deconv_data: 
    imgs[0][0,...,0] = tom_deconv_tomo(**deconv_kwargs)
else:
    pass

# Let's look at the shape of the image
print('Image shape: ', imgs[0].shape)
# The function automatically added two extra dimension to the images:
# One at the front is used to hold a potential stack of images such as a movie.
# One at the end could hold color channels such as RGB.

# Here we extract patches for training and validation.
patch_shape = (96, 96, 96)
patches = datagen.generate_patches_from_list(imgs[:1], shape=patch_shape, shuffle=True)

# Patches are created so they do not overlap.
# (Note: this is not the case if you specify a number of patches. See the docstring for details!)
# Non-overlapping patches enable us to split them into a training and validation set.
n_training = int(0.9*len(patches))
X = patches[:n_training]
X_val = patches[n_training:]

############################## Network configuration ##########################################

epochs = 300

# You can increase "train_steps_per_epoch" to get even better results at the price of longer computation. 
config = N2VConfig(X, unet_kern_size=3, 
                   train_steps_per_epoch=int(X.shape[0]/16),train_epochs=epochs, train_loss='mse', batch_norm=True, 
                   train_batch_size=4, n2v_perc_pix=0.198, n2v_patch_shape=patch_shape, 
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5)


# a name used to identify the model
if use_deconv_data:
    model_name = 'n2v_3D_%s' %name
    # the base directory in which our model will live
    basedir = '/home/ubuntu/Thesis/data/S2SDenoising/n2v_model_logs/%s/deconv/' %name
else:
    model_name = 'n2v_3D_%s' %name
    # the base directory in which our model will live
    basedir = '/home/ubuntu/Thesis/data/S2SDenoising/n2v_model_logs/%s/normal/' %name
# We are now creating our network model.
model = N2V(config=config, name=model_name, basedir=basedir)

############################## Training #############################

history = model.train(X, X_val)
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'])
outfile = os.path.join(basedir, "losses.png")
plt.savefig(outfile, dpi=200)

############################# Prediction #########################

print('Freeing GPU memory before prediction...')
del model 
del config
print('Done!')

model = N2V(config=None, name=model_name, basedir=basedir)

img = read_array(cet_path)
pred = model.predict(img, axes='ZYX')


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
outfile = os.path.join(basedir, "original_vs_N2Vdenoised.png")
plt.savefig(outfile, dpi=200)

if use_deconv_data:
    filename = cet_path.split("/")[-1].replace(".mrc", "_deconv_n2vDenoised")
else:
    filename = cet_path.split("/")[-1].replace(".mrc", "_n2vDenoised")
filename = os.path.join(
    PARENT_PATH, "data/S2SDenoising/denoised/%s.mrc" % (filename)
)

write_array(pred, filename)