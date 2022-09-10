import os
import json


experiment_name = "realBernoulli_dropoutLevel_comparison"
tomogram_list = [
    "tomoPhantom_model8_noisyGaussPoissL",
    "tomoPhantom_model8_noisyGaussPoissM",
    "tomoPhantom_model8_noisyGaussPoissH",
    "tomoPhantom_model14_noisyGaussPoissL",
    "tomoPhantom_model14_noisyGaussPoissM",
    "tomoPhantom_model14_noisyGaussPoissH",
    "tomoPhantom_model16_noisyGaussPoissL",
    "tomoPhantom_model16_noisyGaussPoissM",
    "tomoPhantom_model16_noisyGaussPoissH",
]
deconv_kwargs = {
    "angpix": 14,
    "defocus": 0,
    "snrfalloff": 1,
    "deconvstrength": 1,
    "highpassnyquist": 0.3,
}

max_epochs = 400
experiment_args = {
    "e0": {
        "dataset": "singleCET_dataset",
        "epochs": max_epochs,
        "comment": "Bernoulli",
        "n_bernoulli_samples_prediction": 20,
        "p": 0.1,
    },
    "e1": {
        "dataset": "singleCET_dataset",
        "epochs": max_epochs,
        "comment": "Bernoulli",
        "n_bernoulli_samples_prediction": 20,
        "p": 0.3,
    },
    "e2": {
        "dataset": "singleCET_dataset",
        "epochs": max_epochs,
        "comment": "Bernoulli",
        "n_bernoulli_samples_prediction": 20,
        "p": 0.5,
    },
    "e3": {
        "dataset": "singleCET_dataset",
        "epochs": max_epochs,
        "comment": "Bernoulli",
        "n_bernoulli_samples_prediction": 20,
        "p": 0.7,
    },
}

experiment_logdir = "/home/ubuntu/Thesis/data/S2SDenoising/experiment_args"

default_args = {
    "tomo_name": None,
    "p": 0.3,  # bernoulli masking AND dropout probabilities
    "alpha": 0,
    "n_bernoulli_samples": 6,
    "n_bernoulli_samples_prediction": 6,
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
    "weightedBernoulliMask_prob": 0,
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
