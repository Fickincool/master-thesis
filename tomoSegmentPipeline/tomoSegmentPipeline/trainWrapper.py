from pathlib import Path
import glob
import shutil

DATA_PATH = '/home/haicu/jeronimo.carvajal/Thesis/data'
Isensee_patches_folder = DATA_PATH+'/nnUnet/Task143_cryoET7/'

from tomoSegmentPipeline.training import Train



def make_trainer(dim_in, batch_size, lr, epochs, tb_logdir, model_name, reconstruction_trainer, pretrained_model, Lrnd=18, augment_data=True, test_run=False):
    """Initialize training task using predefined parameters.
    
    """
    if test_run:
        myTrainer = Train(Ncl=2, dim_in=dim_in, lr=lr, weight_decay=0.0, Lrnd=0, tensorboard_logdir=tb_logdir,
         model_name=model_name, pretrained_model=None, augment_data=False)
    else:    
        if reconstruction_trainer:
            print('Selecting trainer for reconstruction task... Ignoring pretrained_model value.')
            raise NotImplementedError
            # myTrainer = TrainPretext(dim_in=dim_in, lr=lr, weight_decay=0.0, tensorboard_logdir=tb_logdir, model_name=model_name, mask_pct=0.3)
        else:
            myTrainer = Train(Ncl=2, dim_in=dim_in, lr=lr, weight_decay=0.0, Lrnd=Lrnd,
             tensorboard_logdir=tb_logdir, model_name=model_name, pretrained_model=pretrained_model, augment_data=augment_data)
    
    myTrainer.batch_size = batch_size
    myTrainer.epochs = epochs

    return myTrainer