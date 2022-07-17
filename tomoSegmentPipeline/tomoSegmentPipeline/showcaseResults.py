from tomoSegmentPipeline.utils.setup import PARENT_PATH

import mrcfile
import numpy as np
import torch
from tomoSegmentPipeline.losses import Tversky_index, Tversky_loss, Tversky1_loss
from tomoSegmentPipeline.model import DeepFinder_model
from tomoSegmentPipeline.utils.common import read_array
from tomoSegmentPipeline.dataloader import (
    tomoSegment_dataset,
    to_categorical,
    transpose_to_channels_first,
)


from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import imageio
import os


def load_model(model_file, Nclass, weight_decay=0):
    "Loads a saved model file by loading the state dictionary."
    aux_train_vals = model_file.split("/")[-1]
    lr = aux_train_vals.split("lr")[-1].split("_")[0]
    lr = float(lr)

    pretrain_type = None  # set this to None for loading, since this parameter is only used for logging training
    model = DeepFinder_model(
        Nclass, Tversky_loss, lr, weight_decay, pretrain_type=pretrain_type
    )
    model.load_state_dict(torch.load(model_file))

    gpu_model = torch.nn.DataParallel(model)
    gpu_model.cuda()

    return gpu_model


def load_tomoData(tomo_file, label_file, is_model_data=True):
    "Load tomogram and corresponding label data."

    # Check we are reading from the same tomogram
    # For running with previous data version
    # tomo_id = tomo_file.split('/')[-1].split('_')[-1]
    # assert tomo_id == label_file.split('/')[-1].split('_')[-1]
    # Version using Isensee data
    if is_model_data:
        tomo_id = tomo_file.split("/")[-1].split(".")[0].replace("_0000", "")
        assert tomo_id == label_file.split("/")[-1].split(".")[0]
    else:
        pass

    tomogram_data = read_array(tomo_file)
    # print("Tomogram shape: ", tomogram_data.shape)

    if label_file is not None:

        lbl_data = read_array(label_file)

    else:
        lbl_data = torch.randint(0, 1, tomogram_data.shape)

    assert tomogram_data.shape == lbl_data.shape

    return tomogram_data, lbl_data


def predict_fullTomogram(tomogram_data, model, dim_in, n_centers, Nclass):
    """Predict a full tomogram.
    
    First load all necessary data, then set up evenly spaced centers along the tomogram to make overlapping patches to make predictions.

    The full tomogram predicted values are the average prediction of the overlapping model outputs for each patch.
    """

    zyx = tomogram_data.shape  # tomogram dimensions
    ref_dim = max(zyx)
    l = int(dim_in / 2)  # size from center

    pcenters = []
    overlaps = []
    for i in zyx:

        factor = i / ref_dim
        new_n_centers = int(np.round(factor * n_centers))
        pcenter = np.linspace(l, i - l, new_n_centers, dtype=int)

        pcenters.append(pcenter)
        overlaps.append(pcenter[0] - pcenter[1] + 2 * l)

    # Assert that the whole tomogram is covered and there is at least some overlapping.
    # Z, Y, X
    if all(np.array(overlaps) > l):
        print(overlaps)
        raise ValueError(
            "There is too much overlap between patches. Reduce the number of centers."
        )

    if not any(np.array(overlaps) > 0):
        print(overlaps)
        raise ValueError(
            "Current patches do not cover the full tomogram. Specify a larger number of centers."
        )

    z, y, x = zyx

    pred_tomo = torch.zeros((1, Nclass, z, y, x)).detach()
    count_tensor = torch.zeros(
        (1, Nclass, z, y, x)
    ).detach()  # for average normalization of overlapping patches

    total_iterations = len(pcenters[0]) * len(pcenters[1]) * len(pcenters[2])

    print("Predicting full tomogram using %i centers..." % n_centers)
    with tqdm(total=total_iterations) as pbar:
        for i in pcenters[0]:
            for j in pcenters[1]:
                for k in pcenters[2]:
                    patch = tomogram_data[i - l : i + l, j - l : j + l, k - l : k + l]
                    patch = torch.as_tensor(patch).unsqueeze(0).unsqueeze(0).to("cuda")

                    # we use normalized patches for training, so we also need to normalize here
                    patch = (patch - torch.mean(patch)) / torch.std(patch)

                    model.eval()
                    with torch.no_grad():
                        pred_patch = model(patch)
                        pred_patch = pred_patch.to("cpu")

                    pred_tomo[
                        :, :, i - l : i + l, j - l : j + l, k - l : k + l
                    ] += pred_patch
                    count_tensor[:, :, i - l : i + l, j - l : j + l, k - l : k + l] += 1
                    pbar.update(1)

    # Get average predictions for overlapping patches
    pred_tomo = pred_tomo / count_tensor

    del count_tensor

    class_pred = pred_tomo[:, 0:2, :, :, :]

    del pred_tomo

    class_pred = class_pred.squeeze().argmax(0).numpy()

    return class_pred


def save_classPred(
    class_pred, model_name, train_tomos, tomo_file, input_type, overwrite=True
):

    class_pred = np.int16(class_pred)

    tomo_id = tomo_file.split("/")[-1].split(".")[0].replace("_0000", "")
    name_prediction_file = "class1Pred_" + tomo_id + ".mrc"
    file_target = PARENT_PATH + "data/processed2/deepFinder/predictions/%s/%s/%s/" % (
        input_type,
        model_name,
        train_tomos,
    )
    Path(file_target).mkdir(parents=True, exist_ok=True)

    file_target += name_prediction_file

    if not Path(file_target).is_file():
        with mrcfile.new(file_target) as mrc:
            mrc.set_data(class_pred)
    elif Path(file_target).is_file() and overwrite:
        with mrcfile.new(file_target, overwrite=overwrite) as mrc:
            mrc.set_data(class_pred)

    return


def load_classPred(model_name, train_tomos, tomo_file, input_type):

    tomo_id = tomo_file.split("/")[-1].split(".")[0].replace("_0000", "")
    name_prediction_file = "class1Pred_" + tomo_id + ".mrc"
    file_target = PARENT_PATH + "data/processed2/deepFinder/predictions/%s/%s/%s/" % (
        input_type,
        model_name,
        train_tomos,
    )
    file_target += name_prediction_file

    if Path(file_target).is_file():
        return read_array(file_target)
    else:
        raise FileNotFoundError


def fullTomogram_modelComparison(
    model_fileList,
    n_centers_list,
    tomo_file,
    label_file,
    overwrite_prediction,
    is_model_data=True,
):

    tomogram_data, classes = load_tomoData(tomo_file, label_file, is_model_data)

    z, y, x = tomogram_data.shape

    classes = torch.tensor(classes)
    # Nclass_data = int(classes.max()+1)
    Nclass_data = 3

    classes = transpose_to_channels_first(
        to_categorical(classes, num_classes=Nclass_data)
    )

    y_true = torch.zeros((1, Nclass_data, z, y, x))
    y_true[0, :, :, :, :] = classes

    weight_decay = 0

    class_predDict = {}
    dice1_dict = {}

    for model_file, n_centers in zip(model_fileList, n_centers_list):
        Nclass_model = 2
        model = load_model(model_file, Nclass_model, weight_decay)
        model_name = model_file.split("/")[-1].replace(".model", "")
        train_tomos = model_file.split("/")[-3]
        input_type = model_file.split("/")[-4]

        key = train_tomos + "/" + model_name

        dim_in = int(model_file.split("in")[-1].split("_")[0])

        if overwrite_prediction:
            class_pred = predict_fullTomogram(
                tomogram_data, model, dim_in, n_centers, Nclass_model
            )
            # print('Saving model predictions...')
            save_classPred(class_pred, model_name, train_tomos, tomo_file, input_type)
        else:
            try:
                class_pred = load_classPred(
                    model_name, train_tomos, tomo_file, input_type
                )
                # print('Found existing tomogram prediction for ', key)
            except FileNotFoundError:
                class_pred = predict_fullTomogram(
                    tomogram_data, model, dim_in, n_centers, Nclass_model
                )
                # print('Saving model predictions...')
                save_classPred(
                    class_pred, model_name, train_tomos, tomo_file, input_type
                )

        class_predDict[key] = class_pred

        y_pred = torch.zeros((1, Nclass_model, z, y, x))
        y_pred[0, :, :, :, :] = transpose_to_channels_first(
            to_categorical(class_pred, Nclass_model)
        )

        Nclass = min(Nclass_data, Nclass_model)

        dice1 = Tversky_index(y_pred, y_true)
        dice1_dict[key] = dice1

    tomo_name = tomo_file.split("/")[-1]

    # if x<200:
    #     _ = write_comparison_gif(class1_predDict, class1, tomogram_data, tomo_name)

    return class_predDict, dice1_dict, classes, tomogram_data


from textwrap import wrap


def make_comparison_plot(class_predDict, classes, tomogram_data):
    pred_keys = list(class_predDict.keys())
    n_models = len(pred_keys)
    n_cols = n_models + 2

    classes = classes.argmax(0)

    for idx_z in np.arange(0, tomogram_data.shape[0], 15):
        fig, axs = plt.subplots(1, n_cols, figsize=(20, 10))

        plt.figure()
        for i in range(n_cols):
            ax = axs[i]
            if i < n_models:
                name = pred_keys[i]
                ax.imshow(class_predDict[name][idx_z])
                ax.set_title("\n".join(wrap(name.replace("_", " "), 20)))
            elif i == n_models:
                name = "Labels"
                ax.imshow(classes[idx_z])
                ax.set_title(name)

            elif i == n_models + 1:
                name = "Observed z = %i" % idx_z
                ax.imshow(tomogram_data[idx_z])
                ax.set_title(name)

        plt.show()

    return


def write_comparison_gif(class_predDict, class1, tomogram_data, tomo_name):
    pred_keys = list(class_predDict.keys())
    n_models = len(pred_keys)
    n_cols = n_models + 2

    save_path = "/home/haicu/jeronimo.carvajal/Thesis/data/processed2/deepFinder/model_comparison/"

    image_files = []

    for idx_z in tqdm(range(tomogram_data.shape[0])):
        # fig, axs = plt.subplots(1, n_cols, figsize=(20, 10))
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        axs = axs.flatten()

        plt.figure()
        for i in range(6):
            ax = axs[i]
            if i < n_models:
                name = pred_keys[i]
                ax.imshow(class_predDict[name][idx_z])
                ax.set_title("\n".join(wrap(name.replace("_", " "), 20)))
            elif i == n_models:
                name = "Labels"
                ax.imshow(class1[idx_z])
                ax.set_title(name)

            elif i == n_models + 1:
                name = "Observed z = %i" % idx_z
                ax.imshow(tomogram_data[idx_z])
                ax.set_title(name)

        filename = save_path + "%s.png" % idx_z

        image_files.append(filename)

        fig.tight_layout()
        fig.suptitle(tomo_name)
        fig.savefig(filename)

        plt.close(fig)

    plt.close()

    with imageio.get_writer(save_path + "%s.gif" % tomo_name, mode="I") as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print("Removing Images\n")
    # Remove files
    for filename in set(image_files):
        os.remove(filename)
    print("DONE")

    return
