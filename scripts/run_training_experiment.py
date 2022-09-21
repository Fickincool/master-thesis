import os
import json


# experiment_name = "realBernoulli_convolved_comparison"
# tomogram_list = ["shrec2021_model4_dummy", "shrec2021_model2_dummy"]
# SHREC21 deconv args
# deconv_kwargs = {
#     "angpix": 10,
#     "defocus": 0,
#     "snrfalloff": 0.3,
#     "deconvstrength": 1,
#     "highpassnyquist": 0.02,
# }
# max_epochs = 400
# experiment_args = {
#     "e0": {
#         "dataset": "singleCET_dataset",
#         "epochs": max_epochs,
#         "comment": "Bernoulli",
#         "n_bernoulli_samples_prediction": 1,
#         "p": 0.3,
#     },
#     "e1": {
#         "dataset": "singleCET_dataset",
#         "epochs": max_epochs,
#         "comment": "Deconvolved Bernoulli",
#         "n_bernoulli_samples_prediction": 1,
#         "p": 0.3,
#         "deconv_kwargs": deconv_kwargs,
#     },
#     "e2": {
#         "dataset": "singleCET_dataset",
#         "epochs": max_epochs,
#         "comment": "Bernoulli",
#         "n_bernoulli_samples_prediction": 1,
#         "p": 0.7,
#     },
#     "e3": {
#         "dataset": "singleCET_dataset",
#         "epochs": max_epochs,
#         "comment": "Deconvolved Bernoulli",
#         "n_bernoulli_samples_prediction": 1,
#         "p": 0.7,
#         "deconv_kwargs": deconv_kwargs,
#     },
# }

# experiment_name = "fourierBernoulli_dropoutLevel_comparison"
# tomogram_list = [
#     "tomoPhantom_model8_noisyGaussPoissVL",
#     "tomoPhantom_model8_noisyGaussPoissL",
#     "tomoPhantom_model8_noisyGaussPoissM",
#     "tomoPhantom_model8_noisyGaussPoissH",
#     "tomoPhantom_model14_noisyGaussPoissVL",
#     "tomoPhantom_model14_noisyGaussPoissL",
#     "tomoPhantom_model14_noisyGaussPoissM",
#     "tomoPhantom_model14_noisyGaussPoissH",
#     # "tomoPhantom_model16_noisyGaussPoissVL",
#     # "tomoPhantom_model16_noisyGaussPoissL",
#     # "tomoPhantom_model16_noisyGaussPoissM",
#     # "tomoPhantom_model16_noisyGaussPoissH",
# ]

# max_epochs = 400
# experiment_args = {
#     "e0": {
#         "dataset": "singleCET_FourierDataset",
#         "epochs": max_epochs,
#         "comment": "Bernoulli",
#         "n_bernoulli_samples_prediction": 1,
#         "p": 0.1,
#     },
#     "e1": {
#         "dataset": "singleCET_FourierDataset",
#         "epochs": max_epochs,
#         "comment": "Bernoulli",
#         "n_bernoulli_samples_prediction": 1,
#         "p": 0.3,
#     },
#     "e2": {
#         "dataset": "singleCET_FourierDataset",
#         "epochs": max_epochs,
#         "comment": "Bernoulli",
#         "n_bernoulli_samples_prediction": 1,
#         "p": 0.5,
#     },
#     "e3": {
#         "dataset": "singleCET_FourierDataset",
#         "epochs": max_epochs,
#         "comment": "Bernoulli",
#         "n_bernoulli_samples_prediction": 1,
#         "p": 0.7,
#     },
# }


experiment_name = "fourierHighFreqMask_comparison"
tomogram_list = [
    "tomoPhantom_model8_noisyGaussPoissVL",
    "tomoPhantom_model8_noisyGaussPoissL",
    "tomoPhantom_model8_noisyGaussPoissM",
    "tomoPhantom_model8_noisyGaussPoissH",
    # "tomoPhantom_model14_noisyGaussPoissVL",
    # "tomoPhantom_model14_noisyGaussPoissL",
    # "tomoPhantom_model14_noisyGaussPoissM",
    # "tomoPhantom_model14_noisyGaussPoissH",
    # "tomoPhantom_model16_noisyGaussPoissVL",
    # "tomoPhantom_model16_noisyGaussPoissL",
    # "tomoPhantom_model16_noisyGaussPoissM",
    # "tomoPhantom_model16_noisyGaussPoissH",
]

max_epochs = 10
experiment_args = {
    "e0": {
        "dataset": "singleCET_FourierDataset",
        "epochs": max_epochs,
        "p":0.7,
        "bernoulliMask_prob": 0,
        "comment": "Fourier with hiFreqMask=1",
        "input_as_target": False,
    },
    "e1": {
        "dataset": "singleCET_FourierDataset",
        "epochs": max_epochs,
        "p":0.7,
        "bernoulliMask_prob": 0.3,
        "comment": "Fourier with hiFreqMask=0.7",
        "input_as_target": False,
    },
    "e2": {
        "dataset": "singleCET_FourierDataset",
        "epochs": max_epochs,
        "p":0.7,
        "bernoulliMask_prob": 0.5,
        "comment": "Fourier with hiFreqMask=0.5",
        "input_as_target": False,
    },
}

experiment_logdir = "/home/ubuntu/Thesis/data/S2SDenoising/experiment_args"

default_args = {
    "tomo_name": None,
    "p": 0.3,  # bernoulli masking AND dropout probabilities
    "alpha": 0,
    "n_bernoulli_samples": 6,
    "total_samples": 100,
    "total_samples_prediction": 150,
    "n_bernoulli_samples_prediction": 1,
    "volumetric_scale_factor": 4,
    "Vmask_probability": 0,
    "Vmask_pct": 0.3,
    "subtomo_length": 96,
    "n_features": 48,
    "batch_size": 2,
    "epochs": 5,
    "lr": 1e-4,
    "num_gpus": 2,
    "dataset": None,
    "predict_simRecon": None,
    "deconv_kwargs": {},
    "use_deconv_as_target": None,
    "comment": None,
    "bernoulliMask_prob": 1,
    "input_as_target": None,
}


def main(experiment_name, args):
    args_str = json.dumps(args)
    print("Training denoising Unet for: %s \n" % args_str)
    os.system("python train_denoisingUnet.py '%s' %s" % (args_str, experiment_name))
    print("\n\n Training finished!!! \n\n")
    os.system("python predict_denoisingUnet.py '%s' %s" % (args_str, experiment_name))


if __name__ == "__main__":

    with open(os.path.join(experiment_logdir, "%s.json" % experiment_name), "w") as f:
        json.dump(experiment_args, f)

    for tomo_name in tomogram_list:
        for exp in experiment_args:
            args = default_args.copy()
            args["tomo_name"] = tomo_name
            # the new args is the dictionary of the experiment arguments
            new_args = experiment_args[exp]

            # rewrite arguments for given experiment
            for arg in new_args:
                args[arg] = new_args[arg]

            # run code
            main(experiment_name, args)
