import os
from cryoS2Sdrop.dataloader import singleCET_dataset
from cryoS2Sdrop.model import Denoising_UNet
from cryoS2Sdrop.losses import self2self_L2Loss

from torch.utils.data import DataLoader
import torch

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

class denoisingTrainer():
    def __init__(self, cet_path, subtomo_length, lr, n_features, p, tensorboard_logdir):
        super().__init__()

        # Hardcoded
        self.loss_fn = self2self_L2Loss()
        self.model = Denoising_UNet(self.loss_fn, lr, n_features, p)

        # model and training stuff
        self.cet_path = cet_path
        self.lr = lr
        self.subtomo_length = subtomo_length  
        self.p = p
        self.n_features = n_features

        # logs
        self.tensorboard_logdir = tensorboard_logdir
        self.model_name = 's2sUNet'

        self.run_init_asserts()

        return

    def run_init_asserts(self):
        if self.subtomo_length%32 != 0:
            raise ValueError('Length of subtomograms must be a multiple of 32 to run the network.')
        return

    def train(self, batch_size, epochs, num_gpus, accelerator="gpu", strategy="ddp"):

        my_dataset = singleCET_dataset(self.cet_path, subtomo_length=self.subtomo_length, p=self.p)

        print('Size of dataset: %i, Steps per epoch: %i. \n' %(len(my_dataset), len(my_dataset)/(batch_size*num_gpus)))

        train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        logger = pl_loggers.TensorBoardLogger(self.tensorboard_logdir, name='',  default_hp_metric=False)

        early_stop_callback = EarlyStopping(monitor='hp/train_loss', min_delta=1e-4, patience=100,
                                            verbose=True, mode='min')

        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [early_stop_callback, lr_monitor]

        trainer = Trainer(logger=logger, log_every_n_steps=1, gpus=num_gpus, max_epochs=epochs,
        enable_progress_bar = False, callbacks=callbacks, accelerator=accelerator, strategy=strategy)

        trainer.fit(self.model, train_loader)

        # if trainer.is_global_zero:
        #     name_model = '%s_ep%i_subtomoLen%i_lr%f.model' %(self.model_name, epochs, self.subtomo_length, self.lr)
        #     self.save_model(name_model)   

        return

    # def save_model(self, name_model):
    #     "Save the trained model in the path_model folder."

    #     # Last version is the one currently being logged
    #     path_model = self.model.logger.log_dir
    #     if not os.path.exists(path_model):
    #         os.mkdir(path_model)
            
    #     name_model = os.path.join(path_model, name_model)
    #     print('Saving model at: ', name_model)
    #     torch.save(self.model.state_dict(), name_model)

    #     return

