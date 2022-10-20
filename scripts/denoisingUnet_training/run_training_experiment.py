import os
import json
from cryoS2Sdrop.make_samples import make_fourier_samples2

#################### 1. Define log directories and input tomograms

# save experiment arguments for this run, for future reproducibility
experiment_logdir = "/home/ubuntu/Thesis/data/example_S2SDenoising/experiment_args"
# here will the tensorboard logs and denoised versions be saved
model_logdir = "/home/ubuntu/Thesis/data/example_S2SDenoising/model_logs/"

experiment_name = "raw_denoising"

# there must be 1 to 1 corresponding in this lists. When no GT is available set gt_list to None
tomogram_list = ["/home/ubuntu/Thesis/data/shrec2021/model_6/reconstruction.mrc"]
# gt is used to compute SSIM and PSNR values during training and for the full tomogram at the end.
gt_list = ["/home/ubuntu/Thesis/data/shrec2021/model_6/grandmodel.mrc"]

################### 2. Default arguments used in every run 

# this arguments will be used when user does not define a value.
# not all of them are used for a given run, some of them are specific to singleCET_FourierDataset,
# some for singleCET_dataset, and some are shared.

# The parameters are for the dataset and also for the training scheme.

default_args = {
    "tomo_name": None, # inferred from path of tomogram as the name of the .mrc file
    "p": 0.3,  # pointwise mask probability **both datasets**
    "dropout_p": 0.3, # dropout prob
    "alpha": 0, # Total variation weight (implemented, not used)
    "n_bernoulli_samples": 6, # number of samples used in a batch **both datasets**
    "total_samples": 100, # total samples for image pool **singleCET_FourierDataset**
    "total_samples_prediction": 150, # total samples (new pool) to use for prediction **singleCET_FourierDataset**
    "n_bernoulli_samples_prediction": 2, # number of samples used for prediction
    "volumetric_scale_factor": 4, # size of the volumetric masks
    "Vmask_probability": 0, # probability of using volumetric mask instead of point mask *singleCET_dataset*
    "Vmask_pct": 0.3, # volumetric mask probability **both datasets**
    "subtomo_length": 96,
    "n_features": 48, # number of features for the Unet
    "batch_size": 2,
    "epochs": 5,
    "lr": 1e-4, # learning rate
    "num_gpus": 2, # multigpu training
    "dataset": None,
    "predict_simRecon": None,
    "deconv_kwargs": {}, # deconvolution kwargs. Empty dict means no deconvolution. Otherwise,
    # a dictionary with all necessary arguments needs to be defined for the convolution. 
    # See deconv_kwargs in 2. Define experiment arguments (below) to see what needs to be included
    "use_deconv_as_target": None,
    "comment": None,
    "bernoulliMask_prob": 1,
    "input_as_target": None,
    "path_to_fourier_samples": None,
    "predict_N_times":100, # number of times to make predictions and then avarge to get final denoised versions
    "clip":True, # wether to clip the inputs or not
}

################## 2. Define experiment arguments
max_epochs = 50
# make pool to be read afterwards (only when images have side larger than 400 px)
make_fourierSamples_beforeTraining = True

# define here deconvolution arguments, uncomment below for Spinach cell tomograms
# deconv_kwargs = {
#     "angpix": 14.08,
#     "defocus": -0.25,
#     "snrfalloff": 0.6,
#     "deconvstrength": 0.45,
#     "highpassnyquist": 0.02,
# }

# form of the experiment_args dict: 
# {experiment_key (not really used, but nice to have):
#  {experiment arguments}
# }
experiment_args = {
    "e0": {
        "dataset": "singleCET_FourierDataset",
        "epochs": max_epochs,
        "p": 0.1,
        "Vmask_pct":0.5,
        "dropout_p":0.8,
        "volumetric_scale_factor":8,
        "comment": "Fourier dropout: 0.8",
        "input_as_target": False,
        "total_samples": 20,
        "total_samples_prediction": 20,
        "n_bernoulli_samples":6,
        "n_bernoulli_samples_prediction":10
    },
}


################ RUN STUFF ###############################
def main(experiment_name, args):
    args_str = json.dumps(args)
    print("Training denoising Unet for: %s \n" % args_str)
    os.system("python train_denoisingUnet.py '%s' %s" % (args_str, experiment_name))
    print("\n\n Training finished!!! \n\n")
    os.system("python predict_denoisingUnet.py '%s' %s" % (args_str, experiment_name))


if __name__ == "__main__":

    with open(os.path.join(experiment_logdir, "%s.json" % experiment_name), "w") as f:
        json.dump(experiment_args, f)

    if gt_list is None:
        gt_list = [None]*len(tomogram_list)

    for cet_path, gt_cet_path in zip(tomogram_list, gt_list):
        for exp in experiment_args:
            args = default_args.copy()
            tomo_name = cet_path.split('/')[-1].replace('.mrc', '')
            args["tomo_name"] = tomo_name
            # the new args is the dictionary of the experiment arguments
            new_args = experiment_args[exp]

            # rewrite arguments for given experiment
            for arg in new_args:
                args[arg] = new_args[arg]

            # add paths to be used for training, logging and prediction:
            args['cet_path'] = cet_path
            args['gt_cet_path'] = gt_cet_path
            args['model_logdir'] = model_logdir

            if "Fourier" in args["dataset"] and make_fourierSamples_beforeTraining:
                samples_path = make_fourier_samples2(args, experiment_name)
                args["path_to_fourier_samples"] = samples_path

            # run code
            main(experiment_name, args)

            # os.system('rm %s' %samples_path)
