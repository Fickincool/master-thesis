import os
import json


################### SETUP ############################################
experiment_name = "raw_denoising_pilot"
tomogram_list = ["tomo04"]

experiment_args = {
    # corresponding version of the model we want to use for prediction
    "e0": {
        "version": "version_0",
        "resample_patch_each_iter": True,
        "total_samples_prediction":30
    }
}
######################################################################

default_args = {
    "tomo_name": None,
    "n_bernoulli_samples_prediction": 1,
    "version": None,
    "resample_patch_each_iter": True,
    "total_samples_prediction": 150,
    "path_to_fourier_samples": None,
    "predict_N_times":250,
    "clip":True,
}


def main(experiment_name, args):
    args_str = json.dumps(args)
    print("Predicting denoising Unet for: %s \n" % args_str)
    os.system("python predict_denoisingUnet.py '%s' %s" % (args_str, experiment_name))


if __name__ == "__main__":
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
