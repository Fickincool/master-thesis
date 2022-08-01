import os
import json


experiment_name = 'denoising_comparison'
experiment_args = {
    'e0': {'dataset':'singleCET_dataset', 'epochs':400},
    'e1': {'dataset':'singleCET_FourierDataset', 'epochs':400},
    'e2': {'dataset':'singleCET_ProjectedDataset', 'batch_size':6, 'epochs':2.3*400},
    }

experiment_logdir = '/home/ubuntu/Thesis/data/S2SDenoising/experiment_args'

default_args = {
    "tomo_name":None,
    "p": 0.3,  # bernoulli masking AND dropout probabilities 
    "alpha": 0,
    "n_bernoulli_samples": 6,
    "volumetric_scale_factor": 4,
    "Vmask_probability": 0,
    "Vmask_pct": 0.3,
    "subtomo_length": 96,
    "n_features": 48,
    "batch_size": 2,
    "epochs": 50,
    "lr": 1e-4,
    "num_gpus": 2,
    "dataset": None
    }


def main(experiment_name, args):
    args_str = json.dumps(args)
    print('Training denoising Unet for: %s \n' %args_str)
    os.system("python train_denoisingUnet.py '%s' %s" %(args_str, experiment_name))

if __name__ == "__main__":

    with open(os.path.join(experiment_logdir, '%s.json' %experiment_name), 'w') as f:
        json.dump(experiment_args, f)

    for exp in experiment_args:
        tomo_name = 'tomo02_cryoCAREDummy_noisyPerlin'
        args = default_args.copy()
        args['tomo_name'] = tomo_name
        # the new args is the dictionary of the experiment arguments
        new_args = experiment_args[exp]

        # rewrite arguments for given experiment
        for arg in new_args:
            args[arg] = new_args[arg] 

        # run code
        main(experiment_name, args)