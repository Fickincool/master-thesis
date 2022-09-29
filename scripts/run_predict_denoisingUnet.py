import os
import json


################### SETUP ############################################
experiment_name = "fourierHFplusVolMask_comparison"
# tomogram_list = [
#     "shrec2021_model4_dummy",
#     "shrec2021_model2_dummy"
# ]

tomogram_list = [
    "tomo04_dummy"
    # "tomoPhantom_model8_noisyGaussPoissVL"
    # "tomoPhantom_model8_noisyGaussPoissL",
    # "tomoPhantom_model8_noisyGaussPoissM",
    # "tomoPhantom_model8_noisyGaussPoissH",
    # "tomoPhantom_model14_noisyGaussPoissL",
    # "tomoPhantom_model14_noisyGaussPoissM",
    # "tomoPhantom_model14_noisyGaussPoissH",
    # "tomoPhantom_model16_noisyGaussPoissL",
    # "tomoPhantom_model16_noisyGaussPoissM",
    # "tomoPhantom_model16_noisyGaussPoissH",
]

experiment_args = {
    # corresponding version of the model we want to use for prediction
    "e0": {"version": "version_5", "n_bernoulli_samples_prediction": 1, "resample_patch_each_iter":True},
    # "e1": {"version": "version_1", "n_bernoulli_samples_prediction": 1, "resample_patch_each_iter":True},
    # "e2": {"version": "version_2", "n_bernoulli_samples_prediction": 1, "resample_patch_each_iter":True},
    # "e3": {"version": "version_3", "n_bernoulli_samples_prediction": 1, "resample_patch_each_iter":True}
}
######################################################################

default_args = {
    "tomo_name": None,
    "n_bernoulli_samples_prediction": 20,
    "version": None,
    "resample_patch_each_iter":True,
    "total_samples_prediction": 150,
    "predict_on_saved_fourier_samples":None
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