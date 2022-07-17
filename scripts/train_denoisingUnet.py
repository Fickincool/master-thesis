import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from tomoSegmentPipeline.utils.common import write_array
from tomoSegmentPipeline.utils import setup
from cryoS2Sdrop.dataloader import singleCET_dataset
from cryoS2Sdrop.trainer import denoisingTrainer
from cryoS2Sdrop.dataloader import randomRotation3D
from cryoS2Sdrop.losses import self2self_L2Loss, self2selfLoss
from cryoS2Sdrop.predict import load_model, predict_full_tomogram

PARENT_PATH = setup.PARENT_PATH

# cet_path = os.path.join(PARENT_PATH, 'data/raw_cryo-ET/tomo02.mrc')
# cet_path = os.path.join(PARENT_PATH, 'data/S2SDenoising/dummy_tomograms/tomo04_deconvDummy.mrc')
cet_path = os.path.join(
    PARENT_PATH, "data/S2SDenoising/dummy_tomograms/tomo02_dummy.mrc"
)


p = 0.3  # dropout probability
n_bernoulli_samples = 18
subtomo_length = 64
n_features = 48

tensorboard_logdir = os.path.join(PARENT_PATH, "data/S2SDenoising/tryout_model_logs")

batch_size = 2
epochs = 30
lr = 1e-4
num_gpus = 2

transform = randomRotation3D(0.5)
loss_fn = self2selfLoss(alpha=0)


s2s_trainer = denoisingTrainer(
    cet_path, subtomo_length, lr, n_features, p, n_bernoulli_samples, tensorboard_logdir, loss_fn
)

s2s_trainer.train(batch_size, epochs, num_gpus, transform=transform)

version = "version_%i" % s2s_trainer.model.logger.version

del s2s_trainer

################### Make prediction plots

torch.cuda.empty_cache()

logdir = os.path.join(tensorboard_logdir, "%s/" % version)

model, hparams = load_model(logdir, DataParallel=True)
my_dataset = singleCET_dataset(cet_path, subtomo_length, p=p)

batch_size = 10
denoised_tomo = []

print("Predicting full tomogram...")
# this is taking two means: first per bernoulli batches, and then again for each time the model was run
# total predictions is the number below*n_bernoulli_samples
for i in tqdm(range(10)):
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


filename = cet_path.split('/')[-1].replace('.mrc', '_s2sDenoised')
v = version.replace('version_', 'v')
filename = os.path.join(PARENT_PATH, 'data/S2SDenoising/denoised/%s_%s.mrc' %(filename, v))

write_array(denoised_tomo.numpy(), filename)