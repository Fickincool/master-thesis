import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from time import sleep
from tomoSegmentPipeline.utils.common import write_array
from tomoSegmentPipeline.utils import setup
from cryoS2Sdrop.dataloader import singleCET_dataset, singleCET_FourierDataset, singleCET_ProjectedDataset
from cryoS2Sdrop.trainer import denoisingTrainer, aggregate_bernoulliSamples, aggregate_bernoulliSamples2, collate_for_oneBernoulliSample
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

args=json.loads(sys.argv[1])
exp_name = sys.argv[2]

p = args['p'] # bernoulli masking probability
n_bernoulli_samples = args['n_bernoulli_samples']
alpha = args['alpha']
volumetric_scale_factor = args['volumetric_scale_factor']
Vmask_probability = args['Vmask_probability']
Vmask_pct = args['Vmask_pct']

subtomo_length = args['subtomo_length'] 
n_features = args['n_features'] 
batch_size = args['batch_size'] 
epochs = args['epochs'] 
lr = args['lr'] 
num_gpus = args['num_gpus'] 
predict_simRecon = args['predict_simRecon']
use_deconv_as_target = args['use_deconv_as_target']

tomo_name = args['tomo_name']

deconv_kwargs = args['deconv_kwargs']

if tomo_name.startswith('model'):
    if tomo_name in ['model14', 'model16']:
        cet_path = os.path.join(
            PARENT_PATH, "data/S2SDenoising/dummy_tomograms/tomoPhantom_%s_Poisson5000+Gauss5+stripes.mrc" %tomo_name
        )
    elif tomo_name == 'model9':
        cet_path = os.path.join(
            PARENT_PATH, "data/S2SDenoising/dummy_tomograms/tomoPhantom_%s_Poisson5000+Gauss5.mrc" %tomo_name
        )

    gt_cet_path = os.path.join(
        PARENT_PATH, "data/S2SDenoising/dummy_tomograms/tomoPhantom_%s.mrc" %tomo_name
    )

elif tomo_name.startswith('tomo'):
    cet_path = os.path.join(
        PARENT_PATH, "data/S2SDenoising/dummy_tomograms/%s.mrc" %tomo_name
    )
    _name = tomo_name.split('_')[0]
    gt_cet_path = os.path.join(
        PARENT_PATH, "/home/ubuntu/Thesis/data/S2SDenoising/dummy_tomograms/%s_cryoCAREDummy.mrc" %_name
    )
    

##################################### Model and dataloader ####################################################

tensorboard_logdir = os.path.join(PARENT_PATH, "data/S2SDenoising/tryout_model_logs/%s/%s/" %(tomo_name, exp_name))
pathlib.Path(tensorboard_logdir).mkdir(parents=True, exist_ok=True) 

comment = args['comment']

n_shift = 10

if args['dataset'] in ['singleCET_FourierDataset', 'singleCET_dataset']:
    _dataset = eval(args['dataset'])
    my_dataset = _dataset(
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
elif args['dataset'] in ['singleCET_ProjectedDataset']:
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

if type(my_dataset)==singleCET_dataset:
    collate_fn = aggregate_bernoulliSamples
    loss_fn = self2selfLoss(alpha=alpha)
    model = Denoising_3DUNet(loss_fn, lr, n_features, p, n_bernoulli_samples)
    model_name = 's2sDenoise3D'
    transform = randomRotation3D(0.5)
    my_dataset.transform = transform

if type(my_dataset)==singleCET_FourierDataset:
    collate_fn = aggregate_bernoulliSamples2
    loss_fn = self2selfLoss_noMask(alpha=alpha)
    model = Denoising_3DUNet_v2(loss_fn, lr, n_features, p, n_bernoulli_samples)
    model_name = 's2sDenoise3D_fourier'
    transform = randomRotation3D_fourierSamples(0.5)

if type(my_dataset)==singleCET_ProjectedDataset:
    collate_fn = collate_for_oneBernoulliSample
    # override bernoulli samples in this case. We always have only one.
    n_bernoulli_samples = 1
    loss_fn = self2selfLoss_noMask(alpha=alpha)
    model = Denoising_3DUNet_v2(loss_fn, lr, n_features, p, n_bernoulli_samples)
    model_name = 's2sDenoise3D_simulatedN2N'
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

# torch.cuda.empty_cache()

# print('Sleeping...')
# sleep(60)
# print('Done!')

# logdir = os.path.join(tensorboard_logdir, "%s/" % version)

# with open(os.path.join(logdir, 'experiment_args.json'), 'w') as f:
#     json.dump(args, f)

# model, hparams = load_model(logdir, DataParallel=True)
# my_dataset.transform = None
# my_dataset.n_shift = 0

# print("Predicting full tomogram...")

# # make a new dataset with more samples
# if args['dataset'] in ['singleCET_FourierDataset', 'singleCET_dataset']:
#     _dataset = eval(args['dataset'])
#     if args['dataset'] == 'singleCET_dataset':
#         n_bernoulli_samples = 20
#     else:
#         n_bernoulli_samples = 12
#     my_dataset = _dataset(
#                 cet_path,
#                 subtomo_length=subtomo_length,
#                 p=p,
#                 n_bernoulli_samples=n_bernoulli_samples,
#                 volumetric_scale_factor=volumetric_scale_factor,
#                 Vmask_probability=Vmask_probability,
#                 Vmask_pct=Vmask_pct,
#                 transform=None,
#                 n_shift=0,
#                 gt_tomo_path=gt_cet_path,
#                 **deconv_kwargs
#             )

# # this is taking two means: first per bernoulli batches, and then again for each time the model was run
# # total predictions is the inner_range*n_bernoulli_samples
# denoised_tomo = predict_full_tomogram(my_dataset, model, N=100)
# print("Done!")

# plt.figure(figsize=(12, 8))
# plt.hist(denoised_tomo.numpy().flatten(), alpha=0.5, label="Denoised")
# plt.hist(my_dataset.data.numpy().flatten(), alpha=0.5, label="Original")
# plt.legend()
# outfile = os.path.join(logdir, "Full_tomo_histograms.png")
# plt.savefig(outfile)

# zidx, yidx, xidx = np.array(my_dataset.data.shape) // 2

# fig, (ax0, ax1) = plt.subplots(2, 3, figsize=(25, 15))
# list(map(lambda axi: axi.set_axis_off(), np.array([ax0, ax1]).ravel()))

# ax0[1].set_title("Original: YX, ZX, ZY")
# ax0[0].imshow(my_dataset.data[zidx])
# ax0[1].imshow(my_dataset.data[:, yidx, :])
# ax0[2].imshow(my_dataset.data[:, :, xidx])

# ax1[1].set_title("Denoised: YX, ZX, ZY")
# ax1[0].imshow(denoised_tomo[zidx])
# ax1[1].imshow(denoised_tomo[:, yidx, :])
# ax1[2].imshow(denoised_tomo[:, :, xidx])

# plt.tight_layout()
# outfile = os.path.join(logdir, "original_vs_denoised.png")
# plt.savefig(outfile, dpi=200)


# ############### Write logs and prediction ########################

# if my_dataset.gt_data is not None:
#     ssim_full = ssim(denoised_tomo.unsqueeze(0), my_dataset.gt_data.unsqueeze(0))
#     ssim_full = float(ssim_full)
#     ssim_baseline = ssim(my_dataset.data.unsqueeze(0), my_dataset.gt_data.unsqueeze(0))
#     ssim_baseline = float(ssim_baseline)

#     psnr_full = peak_signal_noise_ratio(denoised_tomo.unsqueeze(0), my_dataset.gt_data.unsqueeze(0))
#     psnr_full = float(psnr_full)
#     psnr_baseline = peak_signal_noise_ratio(my_dataset.data.unsqueeze(0), my_dataset.gt_data.unsqueeze(0))
#     psnr_baseline = float(psnr_baseline)
    
#     extra_hparams = {
#         "full_tomo_ssim": ssim_full,
#         "full_tomo_psnr": psnr_full,
#         "baseline_ssim": ssim_baseline,
#         "baseline_psnr": psnr_baseline,

#     }
# else:
#     extra_hparams = {
#         "full_tomo_ssim": None,
#         "full_tomo_psnr": None,
#         "baseline_ssim": None,
#         "baseline_psnr": None,
#     }

# sdump = yaml.dump(extra_hparams)
# hparams_file = os.path.join(tensorboard_logdir, version)
# hparams_file = os.path.join(hparams_file, "hparams.yaml")
# with open(hparams_file, "a") as fo:
#     fo.write(sdump)

# filename = cet_path.split("/")[-1].replace(".mrc", "_s2sDenoised")
# filename = os.path.join(
#     logdir, "%s.mrc" % (filename)
# )

# write_array(denoised_tomo.numpy(), filename)
