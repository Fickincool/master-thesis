from cryoS2Sdrop.model import Denoising_3DUNet, Denoising_3DUNet_v2
from cryoS2Sdrop.trainer import aggregate_bernoulliSamples
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from glob import glob
import numpy as np


def aux_forward(model, subtomo):
    with torch.no_grad():
        return model(subtomo)


def load_model(logdir, DataParallel=False):
    "Returns loaded model from checkpoint and hyperparameters"

    with open(glob(logdir + "hparams.yaml")[0]) as f:
        hparams = yaml.load(f, Loader=yaml.BaseLoader)

    # yaml is stupid
    with open(logdir + "hparams.yaml") as f:
        s = f.readlines()
        try:
            dataloader = [x for x in s if 'Dataloader' in x][0]
            dataset = dataloader.split('.')[-1].replace('\n', '').replace('\'', '').strip()
        except:
            dataset = 'Unknown'

    hparams['dataset'] = dataset

    p = float(hparams['p']) # dropout (zeroing) probability
    n_features = int(hparams['n_features'])
    n_bernoulli_samples = int(hparams['n_bernoulli_samples'])

    if dataset in ['singleCET_dataset']:
        model = Denoising_3DUNet(None, 0, n_features, p, n_bernoulli_samples)
        
    if dataset in ['singleCET_FourierDataset', 'singleCET_ProjectedDataset']:
        model = Denoising_3DUNet_v2(None, 0, n_features, p, n_bernoulli_samples)

    ckpt_file = glob(logdir + "checkpoints/*.ckpt")
    assert len(glob(logdir + "checkpoints/*.ckpt")) == 1
    ckpt_file = ckpt_file[0]

    model = model.load_from_checkpoint(ckpt_file).cuda()
    if DataParallel:
        model = torch.nn.DataParallel(model)

    return model, hparams


def collate_fn(batch):
    "Default pytorch collate_fn does not handle None. This ignores None values from the batch."
    batch = [list(filter(lambda x: x is not None, b)) for b in batch]
    return torch.utils.data.dataloader.default_collate(batch)

def predict_full_tomogram(singleCET_dataset, model, N=100):

    tomo_shape = singleCET_dataset.tomo_shape
    subtomo_length = singleCET_dataset.subtomo_length

    denoised_tomo = torch.zeros(tomo_shape)
    count_tensor = torch.zeros(tomo_shape)  # for averaging overlapping patches

    for idx, p0 in enumerate(tqdm(singleCET_dataset.grid)):
        zmin, ymin, xmin = np.array(p0)-subtomo_length//2
        zmax, ymax, xmax = np.array(p0)+subtomo_length//2
        subtomo = singleCET_dataset[idx][0]  # shape: [M, C:=1, S, S, S] or  [C:=1, S, S, S]
        if len(subtomo.shape)==4: # the projected dataset yields this type
            subtomo = subtomo.unsqueeze(0) # equivalent to having "1 Bernoulli sample"
        # we want to average each patch over N Bernoulli samples, typically N >> M
        M = len(subtomo)
        # effective number of samples
        n_times = N//M + 1
        pred = torch.cat([model(subtomo).detach().cpu() for i in range(n_times)])
        # take mean over predictions and reduce channels
        pred = pred.mean(0).squeeze()
        denoised_tomo[zmin:zmax, ymin:ymax, xmin:xmax] += pred
        count_tensor[zmin:zmax, ymin:ymax, xmin:xmax] += 1

    # Get average predictions for overlapping patches
    denoised_tomo = denoised_tomo / count_tensor
    del count_tensor

    return denoised_tomo
