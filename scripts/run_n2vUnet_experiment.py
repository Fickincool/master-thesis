import os
import json


experiment_name = "n2vBaseline"
tomogram_list = [
    "tomoPhantom_model8_noisyGaussPoissVL",
    "tomoPhantom_model14_noisyGaussPoissVL",
    "tomoPhantom_model16_noisyGaussPoissVL",
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
max_epochs = 400
experiment_args = {"e0": {"epochs": max_epochs}}

# tomogram_list = ["shrec2021_model4_dummy", "shrec2021_model2_dummy"]

# # SHREC21 deconv args
# deconv_kwargs = {
#     "angpix": 10,
#     "defocus": 0,
#     "snrfalloff": 0.3,
#     "deconvstrength": 1,
#     "highpassnyquist": 0.02,
# }

# max_epochs = 400
# experiment_args = {
#     "e0": {"epochs": max_epochs},
#     "e1": {"epochs": max_epochs, "deconv_kwargs": deconv_kwargs},
# }

experiment_logdir = "/home/ubuntu/Thesis/data/S2SDenoising/experiment_args"

default_args = {
    "tomo_name": None,
    "deconv_kwargs": {},
    "epochs": 10,
}


def main(experiment_name, args):
    args_str = json.dumps(args)
    print("Training denoising Unet for: %s \n" % args_str)
    os.system("python train_n2vUnet.py '%s' %s" % (args_str, experiment_name))
    print("\n\n Training finished!!! \n\n")


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
