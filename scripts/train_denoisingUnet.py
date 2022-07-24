import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from time import sleep
from tomoSegmentPipeline.utils.common import write_array
from tomoSegmentPipeline.utils import setup
from cryoS2Sdrop.dataloader import singleCET_dataset, singleCET_FourierDataset
from cryoS2Sdrop.trainer import denoisingTrainer, aggregate_bernoulliSamples, aggregate_bernoulliSamples2
from cryoS2Sdrop.dataloader import randomRotation3D, randomRotation3D_fourierSamples
from cryoS2Sdrop.losses import self2self_L2Loss, self2selfLoss, self2selfLoss_noMask
from cryoS2Sdrop.model import Denoising_3DUNet, Denoising_3DUNet_v2
from cryoS2Sdrop.predict import load_model, predict_full_tomogram
from pytorch_msssim import ssim
from torchmetrics.functional import peak_signal_noise_ratio

PARENT_PATH = setup.PARENT_PATH

###################### Input data definition ###################

# cet_path = os.path.join(PARENT_PATH, 'data/raw_cryo-ET/tomo02.mrc')
# cet_path = os.path.join(PARENT_PATH, 'data/S2SDenoising/dummy_tomograms/tomo04_deconvDummy.mrc')
tomo_name = 'tomo02_dummy'
cet_path = os.path.join(
    PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s.mrc" %tomo_name
)

gt_cet_path = None

# simulated_model = 'model16'
# simulated_model = 'model14'
# cet_path = os.path.join(
#     PARENT_PATH, "data/S2SDenoising/dummy_tomograms/tomoPhantom_%s_Poisson5000+Gauss5+stripes.mrc" %simulated_model
# )

# simulated_model = 'model9'
# cet_path = os.path.join(
#     PARENT_PATH, "data/S2SDenoising/dummy_tomograms/tomoPhantom_%s_Poisson5000+Gauss5.mrc" %simulated_model
# )

# gt_cet_path = os.path.join(
#     PARENT_PATH, "data/S2SDenoising/dummy_tomograms/tomoPhantom_%s.mrc" %simulated_model
# )

##################################### Model and dataloader ####################################################
p = 0.3  # dropout probability
n_bernoulli_samples = 6
volumetric_scale_factor = 4
Vmask_probability = 0
Vmask_pct = 0.3

subtomo_length = 96
n_features = 48
try:
    name = simulated_model
except:
    name = tomo_name

tensorboard_logdir = os.path.join(PARENT_PATH, "data/S2SDenoising/tryout_model_logs/%s/" %name)
comment = 'Dummy tomo02 Fourier sample version'

batch_size = 2
epochs = 400
lr = 1e-4
num_gpus = 2

my_dataset = singleCET_FourierDataset(
            cet_path,
            subtomo_length=subtomo_length,
            p=p,
            n_bernoulli_samples=n_bernoulli_samples,
            volumetric_scale_factor=volumetric_scale_factor,
            Vmask_probability=Vmask_probability,
            Vmask_pct=Vmask_pct,
            transform=None,
            gt_tomo_path=gt_cet_path
        )

if type(my_dataset)==singleCET_dataset:
    collate_fn = aggregate_bernoulliSamples
    loss_fn = self2selfLoss(alpha=0)
    model = Denoising_3DUNet(loss_fn, lr, n_features, p, n_bernoulli_samples)
    model_name = 's2sDenoise3D'
    transform = randomRotation3D(0.5)
    my_dataset.transform = transform

if type(my_dataset)==singleCET_FourierDataset:
    collate_fn = aggregate_bernoulliSamples2
    loss_fn = self2selfLoss_noMask(alpha=1e-5)
    model = Denoising_3DUNet_v2(loss_fn, lr, n_features, 0.3, n_bernoulli_samples)
    model_name = 's2sDenoise3D_fourier'
    transform = randomRotation3D_fourierSamples(0.5)


##################################### Training ####################################################

s2s_trainer = denoisingTrainer(
    model,
    my_dataset,
    tensorboard_logdir,
    model_name=model_name
)

s2s_trainer.train(collate_fn, batch_size, epochs, num_gpus, transform=transform, comment=comment)

version = "version_%i" % s2s_trainer.model.logger.version

del s2s_trainer

################### Make prediction plots ########################

torch.cuda.empty_cache()

print('Sleeping...')
sleep(60)
print('Done!')

logdir = os.path.join(tensorboard_logdir, "%s/" % version)

model, hparams = load_model(logdir, model, DataParallel=True)
my_dataset.transform = None

batch_size = 10
denoised_tomo = []

print("Predicting full tomogram...")
# this is taking two means: first per bernoulli batches, and then again for each time the model was run
# total predictions is the inner_range*n_bernoulli_samples
total_preds = 100
inner_range = total_preds//n_bernoulli_samples
for i in tqdm(range(inner_range)):
    _denoised_tomo = predict_full_tomogram(my_dataset, model, batch_size)
    denoised_tomo.append(_denoised_tomo)

denoised_tomo = torch.stack(denoised_tomo).mean(0)
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


############### Write logs and prediction ########################

if my_dataset.gt_data is not None:
    ssim_full = ssim(denoised_tomo.unsqueeze(0), my_dataset.gt_data.unsqueeze(0))
    ssim_full = float(ssim_full)

    psnr_full = peak_signal_noise_ratio(denoised_tomo.unsqueeze(0), my_dataset.gt_data.unsqueeze(0))
    psnr_full = float(psnr_full)
    
    extra_hparams = {
        "full_tomo_ssim": ssim_full,
        "full_tomo_psnr": psnr_full,
    }
else:
    extra_hparams = {
        "full_tomo_ssim": None,
        "full_tomo_psnr": None,
    }

sdump = yaml.dump(extra_hparams)
hparams_file = os.path.join(tensorboard_logdir, version)
hparams_file = os.path.join(hparams_file, "hparams.yaml")
with open(hparams_file, "a") as fo:
    fo.write(sdump)

filename = cet_path.split("/")[-1].replace(".mrc", "_s2sDenoised")
v = version.replace("version_", "v")
filename = os.path.join(
    PARENT_PATH, "data/S2SDenoising/denoised/%s_%s.mrc" % (filename, v)
)

write_array(denoised_tomo.numpy(), filename)
