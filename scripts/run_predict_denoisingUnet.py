import os
import json


################### SETUP ############################################
experiment_name = "fourierWeightedBernoulliMask_comparison"
tomo_name = "tomo02_dummy"
experiment_args = {
    # corresponding version of the model we want to use for prediction
    "e2": {"version": "version_1", "n_bernoulli_samples": 12}
}
######################################################################

default_args = {"tomo_name": None, "n_bernoulli_samples": 20, "version": None}


def main(experiment_name, args):
    args_str = json.dumps(args)
    print("Predicting denoising Unet for: %s \n" % args_str)
    os.system("python predict_denoisingUnet.py '%s' %s" % (args_str, experiment_name))


if __name__ == "__main__":

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
