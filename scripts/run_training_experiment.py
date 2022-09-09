import os
import json


experiment_name = "fourierWeightedBernoulliMask_comparison"
tomo_name = "tomo02_dummy"
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
        "dataset": "singleCET_FourierDataset",
        "epochs": 400,
        "weightedBernoulliMask_prob": 1,
        "comment": "Fourier with weightedBernoulliMask on raw data",
        "input_as_target": False,
    },
    "e1": {
        "dataset": "singleCET_FourierDataset",
        "epochs": 400,
        "weightedBernoulliMask_prob": 0.5,
        "comment": "Fourier with weightedBernoulliMask=0.5 on raw data",
        "input_as_target": False,
    },
    "e2": {
        "dataset": "singleCET_FourierDataset",
        "epochs": 400,
        "weightedBernoulliMask_prob": 0,
        "comment": "Fourier no weightedBernoulliMask (OG) on raw data",
        "input_as_target": False,
    }
}

experiment_logdir = "/home/ubuntu/Thesis/data/S2SDenoising/experiment_args"

default_args = {
    "tomo_name": None,
    "p": 0.3,  # bernoulli masking AND dropout probabilities
    "alpha": 0,
    "n_bernoulli_samples": 6,
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


if __name__ == "__main__":

    with open(os.path.join(experiment_logdir, "%s.json" % experiment_name), "w") as f:
        json.dump(experiment_args, f)

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
