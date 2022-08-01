import os
import json


experiment_name = 'rawTomo_denoising_comparison'
aux_epochs = 3
experiment_args = {
    # corresponding version of the model we want to use for prediction
    'e0': {'version':'version_1', 'n_bernoulli_samples':16}
    }

default_args = {
    "tomo_name":None,
    "n_bernoulli_samples":30,
    "version": None
    }


def main(experiment_name, args):
    args_str = json.dumps(args)
    print('Predicting denoising Unet for: %s \n' %args_str)
    os.system("python predict_denoisingUnet.py '%s' %s" %(args_str, experiment_name))

if __name__ == "__main__":

    for exp in experiment_args:
        tomo_name = 'tomo02_dummy'
        args = default_args.copy()
        args['tomo_name'] = tomo_name
        # the new args is the dictionary of the experiment arguments
        new_args = experiment_args[exp]

        # rewrite arguments for given experiment
        for arg in new_args:
            args[arg] = new_args[arg] 

        # run code
        main(experiment_name, args)