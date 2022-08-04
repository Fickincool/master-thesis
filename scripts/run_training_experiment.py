import os
import json


experiment_name = 'normal_vs_deconvolved_comparison'
deconv_kwargs = {
    'angpix': 14,
    'defocus': 0,
    'snrfalloff': 1,
    'deconvstrength': 1,
    'highpassnyquist': 0.3
                 }

max_epochs = 400
experiment_args = {
    'e0': {'dataset':'singleCET_dataset','deconv_kwargs':deconv_kwargs, 'epochs':max_epochs, 'comment':"Deconvolved bernoulli"},
    'e1': {'dataset':'singleCET_dataset', 'epochs':max_epochs, 'comment':"Bernoull"},

    'e2': {'dataset':'singleCET_FourierDataset','deconv_kwargs':deconv_kwargs, 'epochs':max_epochs, 'comment': 'Deconvolved Fourier'},
    'e3': {'dataset':'singleCET_FourierDataset', 'epochs':max_epochs, 'comment': 'Fourier'},
    
    'e4': {
        'dataset':'singleCET_ProjectedDataset', 'deconv_kwargs':deconv_kwargs, 'epochs':2.3*max_epochs,
        'batch_size':6, 'use_deconv_as_target': True, 'predict_simRecon': True,
        'comment':"Sim N2N denoising using deconv reconstruction as target."
        },
    'e5': {
        'dataset':'singleCET_ProjectedDataset', 'deconv_kwargs':deconv_kwargs, 'epochs':2.3*max_epochs,
        'batch_size':6, 'use_deconv_as_target': False, 'predict_simRecon': True,
        'comment':"Sim N2N denoising using reconstruction as target."
        },

    'e6': {
        'dataset':'singleCET_ProjectedDataset', 'deconv_kwargs':deconv_kwargs, 'epochs':2.3*max_epochs,
        'batch_size':6, 'use_deconv_as_target': True, 'predict_simRecon': False,
        'comment':"Sim N2N denoising using deconv data as target."
        },
    'e7': {
        'dataset':'singleCET_ProjectedDataset', 'deconv_kwargs':deconv_kwargs, 'epochs':2.3*max_epochs,
        'batch_size':6, 'use_deconv_as_target': False, 'predict_simRecon': False,
        'comment':"Sim N2N denoising using raw data as target."
        },

        
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
    "epochs": 5,
    "lr": 1e-4,
    "num_gpus": 2,
    "dataset": None,
    "predict_simRecon": None,
    "deconv_kwargs": {},
    "use_deconv_as_target": None,
    "comment": None
    }


def main(experiment_name, args):
    args_str = json.dumps(args)
    print('Training denoising Unet for: %s \n' %args_str)
    os.system("python train_denoisingUnet.py '%s' %s" %(args_str, experiment_name))

if __name__ == "__main__":

    with open(os.path.join(experiment_logdir, '%s.json' %experiment_name), 'w') as f:
        json.dump(experiment_args, f)

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