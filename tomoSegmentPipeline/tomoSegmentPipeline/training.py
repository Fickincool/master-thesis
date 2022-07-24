# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (serpico team); adapted by Lorenz Lamm
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================


from gc import callbacks
from pyexpat import model
import h5py
import numpy as np
import time
import dill as pickle

import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pathlib import Path
from glob import glob
import random
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


from pytorch_lightning import loggers as pl_loggers
from tomoSegmentPipeline.dataloader import tomoSegment_dataset, tomoSegment_dummyDataset
from tomoSegmentPipeline import model
from tomoSegmentPipeline import losses
from tomoSegmentPipeline.utils import core, setup

from tomoSegmentPipeline.utils.setup import PARENT_PATH


# modelSummary_folder = PARENT_PATH+'models/model_summary'

# TODO: add method for resuming training. It should load existing weights and train_history. So when restarting, the plot curves show prececedent epochs
class Train(core.DeepFinder):
    def __init__(
        self,
        Ncl,
        dim_in,
        lr,
        weight_decay,
        Lrnd,
        tensorboard_logdir,
        model_name,
        augment_data,
        batch_size,
        epochs,
        pretrained_model=None,
    ):
        core.DeepFinder.__init__(self)
        self.tensorboard_logdir = tensorboard_logdir

        # Network parameters:
        self.Ncl = Ncl  # Ncl
        self.dim_in = dim_in  # /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        # self.net = models.my_model(self.dim_in, self.Ncl)
        # self.loss_fn = losses.Tversky1_loss() # Tversky1_loss considers only class 1
        self.loss_fn = losses.Tversky_loss()
        self.lr = lr
        self.Lrnd = Lrnd  # random shifts applied when sampling data- and target-patches (in voxels)

        self.augment_data = augment_data
        self.pretrained_model = pretrained_model

        self.pretrain_type = self.get_pretrain_type()

        self.model = model.DeepFinder_model(
            self.Ncl, self.loss_fn, lr, weight_decay, self.pretrain_type
        )

        self.model_name = model_name

        self.initialize_pretrained_model()

        # Training parameters:
        self.batch_size = batch_size
        self.epochs = epochs

        self.check_attributes()

    def get_pretrain_type(self):

        if self.pretrained_model is None:
            return

        elif "proxyLabel" in self.pretrained_model:
            return "proxyLabel"

        elif "reconstruction" in self.pretrained_model:
            return "reconstructionTask"

        else:
            raise ValueError(
                "Pretrain type can only be: proxyLabel, reconstructionTask or None"
            )

    def initialize_pretrained_model(self):

        if self.pretrain_type is None:
            pass

        elif self.pretrain_type == "proxyLabel":
            self.model.load_state_dict(torch.load(self.pretrained_model))

        elif self.pretrain_type == "reconstructionTask":
            pretrained_dict = torch.load(self.pretrained_model)
            model_dict = self.model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k not in ["layer5.4.weight", "layer5.4.bias"]
            }
            # Use regular initialization for the last layer, which are different in the reconstruction task
            pretrained_dict["layer5.4.weight"] = model_dict["layer5.4.weight"]
            pretrained_dict["layer5.4.bias"] = model_dict["layer5.4.bias"]

            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)

            # 3. load the new state dict
            self.model.load_state_dict(pretrained_dict)

        return

    def check_attributes(self):
        self.is_positive_int(self.Ncl, "Ncl")
        self.is_multiple_4_int(self.dim_in, "dim_in")
        self.is_positive_int(self.batch_size, "batch_size")
        self.is_positive_int(self.epochs, "epochs")
        self.is_int(self.Lrnd, "Lrnd")

    def check_arguments(self, path_data, path_target):
        self.is_list(path_data, "path_data")
        self.is_list(path_target, "path_target")
        self.are_lists_same_length(
            [path_data, path_target], ["data_list", "target_list"]
        )

    def launch(
        self,
        train_tomos,
        val_tomos,
        input_type,
        num_gpus,
        accelerator,
        num_workers,
        nPatches_training=None,
        resume_from_checkpoint=None,
        train_callbacks=None,
        dataset=tomoSegment_dataset,
    ):
        """

        """
        self.check_attributes()

        # we only need to modify this to change the type of input data
        paths_trainData, paths_trainTarget = setup.get_paths(train_tomos, input_type)
        paths_valData, paths_valTarget = setup.get_paths(val_tomos, input_type)

        if nPatches_training is not None:
            # training with tomo02, 03 and 17 yields 32 patches in total.
            assert nPatches_training <= len(paths_trainData)
            # this guarantees that we always use the same incremental list of random patches
            random.seed(17)
            random_indices = random.sample(
                range(len(paths_trainData)), len(paths_trainData)
            )[0:nPatches_training]

            print("Random indices used for training: ", random_indices)
            paths_trainData = list(np.array(paths_trainData)[random_indices])
            paths_trainTarget = list(np.array(paths_trainTarget)[random_indices])

        self.check_arguments(paths_trainData, paths_trainTarget)
        self.check_arguments(paths_valData, paths_valTarget)

        if dataset == tomoSegment_dataset:
            train_set = dataset(
                paths_trainData,
                paths_trainTarget,
                self.dim_in,
                self.Ncl,
                self.Lrnd,
                self.augment_data,
            )
            # Note: augmentation is off for the validation set
            val_set = dataset(
                paths_valData, paths_valTarget, self.dim_in, self.Ncl, 0, False
            )
        elif dataset == tomoSegment_dummyDataset:
            print("Running training to overfit one patch...")
            train_set = dataset(self.dim_in, self.Ncl)
            val_set = dataset(self.dim_in, self.Ncl, paths_valData, paths_valTarget)

        # print(len(train_set))

        # from the docs: It is generally not recommended to return CUDA tensors in multi-process loading
        # because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing
        # (see CUDA in multiprocessing). Instead, we recommend using automatic memory pinning (i.e., setting pin_memory=True),
        #  which enables fast data transfer to CUDA-enabled GPUs.
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )

        logger = pl_loggers.TensorBoardLogger(
            self.tensorboard_logdir, name="", default_hp_metric=False
        )

        trainer = pl.Trainer(
            logger=logger,
            log_every_n_steps=1,
            gpus=num_gpus,
            max_epochs=self.epochs,
            enable_progress_bar=False,
            accelerator=accelerator,
            resume_from_checkpoint=resume_from_checkpoint,
            callbacks=train_callbacks,
        )

        # maybe add training time here
        trainer.fit(self.model, train_loader, val_loader)

        train_patchCounts = [x.split("/")[-1].split("_")[0] for x in paths_trainData]
        train_patchCounts = np.unique(train_patchCounts, return_counts=True)
        train_patchCounts = dict(zip(*train_patchCounts))

        val_patchCounts = [x.split("/")[-1].split("_")[0] for x in paths_valData]
        val_patchCounts = np.unique(val_patchCounts, return_counts=True)
        val_patchCounts = dict(zip(*val_patchCounts))

        if trainer.is_global_zero:
            self.save_model()

            modelSummary_folder = self.tensorboard_logdir.split("/logs")[0]
            self.write_modelSummary(
                modelSummary_folder, train_patchCounts, val_patchCounts, input_type
            )

        return trainer

    def save_model(self):
        "Save the trained model in the path_model folder."
        # Last version is the one currently being logged
        version = self.model.logger.version

        path_model = self.model.logger.log_dir
        if not os.path.exists(path_model):
            os.mkdir(path_model)

        name_model = "%s_ep%i_in%i_lr%f_%s.model" % (
            self.model_name,
            self.epochs,
            self.dim_in,
            self.lr,
            version,
        )
        name_model = os.path.join(path_model, name_model)
        print("Saving model at: ", name_model)
        torch.save(self.model.state_dict(), name_model)

        trainer_name = os.path.join(
            path_model, name_model.replace(".model", "_trainer.pkl")
        )

        with open(trainer_name, "wb") as file:
            pickle.dump(self, file)

        return

    def write_modelSummary_row(self, modelSummary_folder, row_data):
        "Write a row of the model summary corresponding to the current model."

        header = """Name,Logs,Input type,Training set,Validation set,Data Augmentation,Random Shifts,Learning rate,Batch size,Patch size,Training epochs,Pretrained type,Pretrained model path,Loss function,Best epoch validation Loss,Train loss epoch (From best validation epoch),Class 1 Dice score epoch (From best validation epoch),Best validation epoch,Comment"""

        if not Path(modelSummary_folder).is_dir():
            print("Creating modelSummary folder...")
            os.mkdir(modelSummary_folder)

        print("Opening ModelSummary file...")
        try:
            file = open(modelSummary_folder + "/model_summary.csv", "r+")
            print("ModelSummary file exists, appending current model data...")
            # check header
            first_line = next(file).replace("\n", "")
            assert first_line == header.replace("\n", "")

            file.writelines("\n")
            file.writelines("\n" + row_data)

            print("New line written! Closing summary file...")

            file.close()

        except IOError:
            print("No file found, writing header and row_data...")
            file = open(modelSummary_folder + "/model_summary.csv", "a+")
            file.writelines([header.replace("\n", ""), "\n" + row_data])

            file.close()

        return

    def write_modelSummary(
        self, modelSummary_folder, train_set, validation_set, input_type
    ):
        """
        Write to the model summary csv file in the modelSummary_folder.
        If the modelSummary file doesn't exists, create it. Otherwise it appends the row with the current model data to it.
        """

        print("\nWriting to modelSummary...")

        # Getting the logged values directly ensures we get the synced values through all GPUs
        logdir_path = self.model.logger.log_dir

        events_path = glob(os.path.join(logdir_path, "events.*"))[0]
        event_acc = EventAccumulator(events_path)
        event_acc.Reload()

        _, _, values_valLoss = zip(*event_acc.Scalars("hp/val_loss"))
        best_val_loss_epoch = np.min(values_valLoss)
        best_val_loss_epoch_idx = np.argmin(values_valLoss)  # index starts count at 0

        effective_epochs = len(values_valLoss)

        # _, _, values_dice = zip(*event_acc.Scalars('hp/val_dice_epoch'))
        _, _, values_trainLoss = zip(*event_acc.Scalars("hp/train_loss_epoch"))

        # associated_val_class1_dice = float(values_dice[best_val_loss_epoch_idx])
        associated_val_class1_dice = -1
        associated_train_loss_epoch = float(values_trainLoss[best_val_loss_epoch_idx])

        train_set = str(train_set)[1:-1].replace(",", " ")
        train_set = train_set.replace(",", " -").replace("'", "")

        validation_set = str(validation_set)[1:-1].replace(",", " ")
        validation_set = validation_set.replace(",", " -").replace("'", "")

        epochs_str = "%i out of %i" % (effective_epochs, self.epochs)

        row_data = [
            self.model_name,
            logdir_path,
            input_type,
            train_set,
            validation_set,
            self.augment_data,
            self.Lrnd,
            self.model.lr,
            self.batch_size,
            self.dim_in,
            epochs_str,
            self.pretrain_type,
            self.pretrained_model,
            self.loss_fn,
            best_val_loss_epoch,
            associated_train_loss_epoch,
            associated_val_class1_dice,
            best_val_loss_epoch_idx,
        ]

        row_data = ",".join([str(x) for x in row_data])
        self.write_modelSummary_row(modelSummary_folder, row_data)

        return


# class TrainPretext(Train):
#     """
#     Class inherited from the Train class of the DeepFinder model.

#     The pretext task is image inpainting of random squares within the input 3D tomograms.
#     """
#     def __init__(self, dim_in, lr, weight_decay, tensorboard_logdir, model_name, mask_pct):
#         Train.__init__(self, Ncl=1, dim_in=dim_in, lr=lr, weight_decay=weight_decay, Lrnd=0,
#         tensorboard_logdir=tensorboard_logdir, model_name=model_name, pretrained_model=None)

#         self.loss_fn = torch.nn.MSELoss()
#         self.mask_pct = mask_pct
#         self.model = model.PretextDeepFinder_model(self.loss_fn, lr, weight_decay)

#     # see Train class
#     def launch(self, path_data, path_target, objlist_train, objlist_valid):
#         """This function launches the training procedure. For each epoch, an image is plotted, displaying the progression
#         with different metrics: loss, accuracy, f1-score, recall, precision. Every 10 epochs, the current network weights
#         are saved.

#         Args:
#             path_data (list of string): contains paths to data files (i.e. tomograms)
#             path_target (list of string): contains paths to target files (i.e. annotated volumes)
#             objlist_train (list of dictionaries): contains information about annotated objects (e.g. class, position)
#                 In particular, the tomo_idx should correspond to the index of 'path_data' and 'path_target'.
#                 See utils/objl.py for more info about object lists.
#                 During training, these coordinates are used for guiding the patch sampling procedure.
#             objlist_valid (list of dictionaries): same as 'objlist_train', but objects contained in this list are not
#                 used for training, but for validation. It allows to monitor the training and check for over/under-fitting.
#                 Ideally, the validation objects should originate from different tomograms than training objects.

#         Note:
#             The function saves following files at regular intervals:
#                 net_weights_epoch*.h5: contains current network weights

#                 net_train_history.h5: contains arrays with all metrics per training iteration

#                 net_train_history_plot.png: plotted metric curves

#         """
#         self.check_attributes()
#         self.check_arguments(path_data, path_target, objlist_train, objlist_valid)


#         train_set = PretextDeepFinder_dataset(self.flag_direct_read, path_data, path_target, objlist_train,
#          self.dim_in, self.Lrnd, self.h5_dset_name, self.mask_pct)
#         train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

#         val_set = PretextDeepFinder_dataset(self.flag_direct_read, path_data, path_target, objlist_valid,
#          self.dim_in, self.Lrnd, self.h5_dset_name, self.mask_pct)
#         val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)


#         logger = pl_loggers.TensorBoardLogger(self.tensorboard_logdir)

#         # this runs on supergpu03pxe
#         num_gpus = torch.cuda.device_count()
#         if num_gpus>1:
#             accelerator = 'dp'
#         else:
#             None

#         trainer = pl.Trainer(logger=logger, log_every_n_steps=1, gpus=num_gpus, max_epochs=self.epochs, progress_bar_refresh_rate=0,
#         accelerator=accelerator)
#         trainer.fit(self.model, train_loader, val_loader)

#         self.save_model()

#         self.write_modelSummary(modelSummary_folder, path_data, path_target, objlist_train, objlist_valid)
