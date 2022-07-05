from cryoS2Sdrop.model import Denoising_UNet
import torch
from tqdm import tqdm



def load_model(ckpt_file):
    "Load a model from checkpoint and send to cuda."
    
    model = Denoising_UNet.load_from_checkpoint(ckpt_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    gpu_model = torch.nn.DataParallel(model)
    gpu_model.to(device)
    
    return gpu_model

def predict_full_tomogram(singleCET_dataset, model):

    tomo_shape = singleCET_dataset.tomo_shape
    subtomo_length = singleCET_dataset.subtomo_length

    denoised_tomo = torch.zeros(tomo_shape)
    count_tensor = torch.zeros(tomo_shape) # for averaging overlapping patches

    for idx, (subtomo, target, _) in enumerate(tqdm(singleCET_dataset)):
        
        subtomo, target = subtomo.unsqueeze(0), target.unsqueeze(0)
        
        with torch.no_grad():
            denoised_subtomo = model(subtomo).squeeze().mean(axis=0).cpu()
        
        z0, y0, x0 = singleCET_dataset.grid[idx]
        zmin, zmax = z0-subtomo_length//2, z0+subtomo_length//2
        ymin, ymax = y0-subtomo_length//2, y0+subtomo_length//2
        xmin, xmax = x0-subtomo_length//2, x0+subtomo_length//2
        
        count_tensor[zmin:zmax, ymin:ymax, xmin:xmax] += 1
        denoised_tomo[zmin:zmax, ymin:ymax, xmin:xmax] += denoised_subtomo
        
    # Get average predictions for overlapping patches
    denoised_tomo = denoised_tomo/count_tensor

    del count_tensor

    return denoised_tomo