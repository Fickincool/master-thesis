from multiprocessing.sharedctypes import Value
import torch
import numpy as np
from torch.utils.data import Dataset
from tomoSegmentPipeline.utils.common import read_array

class singleCET_dataset(Dataset):
    def __init__(self, tomo_path, subtomo_length, p, n_samples, volumetric_scale_factor=8, transform=None):  
        """
        Load cryoET dataset for self2self denoising.

        The dataset consists of subtomograms of shape [n_samples, s, s, s] 
        where n_samples is the number of Bernoulli samples and s is the subtomogram side length.

        - tomo_path: tomogram path
        - subtomo_length: side length of the patches to be used for training
        - n_samples: number of independent bernoulli samples
        - p: probability of an element to be zeroed
        - volumetric_scale_factor: times the original tomogram shape will be reduced 
        to take bernoulli point samples before upsampling into volumetric bernoulli blind spots.
        """
        self.tomo_path = tomo_path
        self.data = torch.tensor(read_array(tomo_path))
        self.data = self.clip(self.data)
        self.data = self.normalize(self.data)
        self.tomo_shape = self.data.shape
        self.subtomo_length = subtomo_length
        self.grid = self.create_grid()
        self.transform = transform # think how to implement this
        self.p = p
        self.dropout = torch.nn.Dropout(p=p)
        self.upsample = torch.nn.Upsample(scale_factor=volumetric_scale_factor)
        self.n_samples = n_samples
        self.vol_scale_factor = volumetric_scale_factor

        self.run_init_asserts()

        return

    def run_init_asserts(self):
        if self.subtomo_length%self.vol_scale_factor != 0:
            raise ValueError('Length of subtomograms must a multiple of the volumetric scale factor.')

        return

    def normalize(self, X:torch.tensor):
        mean = X.mean()
        std = X.std()
        normalized_data = (X - mean) / std
        return normalized_data

    def clip(self, X, low=0.005, high=0.995):
        # works with tensors =)
        return np.clip(X, np.quantile(X, low), np.quantile(X, high))

    def __len__(self):
        return len(self.grid)

    def get_volumetric_blind_spots(self):
        downsampled_shape = np.array(3*[self.subtomo_length])//self.vol_scale_factor
        downsampled_shape = tuple(downsampled_shape)

        bernoulli_Vmask = torch.stack([self.dropout(torch.ones(downsampled_shape))*(1-self.p) 
        for i in range(self.n_samples)], axis=0)
        bernoulli_Vmask = bernoulli_Vmask.unsqueeze(0)
        bernoulli_Vmask = self.upsample(bernoulli_Vmask).squeeze()
        return bernoulli_Vmask

    def __getitem__(self, index:int):
        center_z, center_y, center_x = self.grid[index]
        z_min, z_max = center_z-self.subtomo_length//2, center_z+self.subtomo_length//2
        y_min, y_max = center_y-self.subtomo_length//2, center_y+self.subtomo_length//2
        x_min, x_max = center_x-self.subtomo_length//2, center_x+self.subtomo_length//2
        subtomo = self.data[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # first transform and then get samples
        if self.transform:
            subtomo = self.transform(subtomo)

        # bernoulli mask with no power correction from the dropout
        # bernoulli_mask = torch.stack([self.dropout(torch.ones_like(subtomo))*(1-self.p) for i in range(self.n_samples)], axis=0)
        bernoulli_mask = self.get_volumetric_blind_spots()

        _samples = subtomo.unsqueeze(0).repeat(self.n_samples, 1, 1, 1) # get n samples
        bernoulli_subtomo = bernoulli_mask*_samples  # bernoulli samples
        target = (1-bernoulli_mask)*_samples # complement of the bernoulli sample

        return bernoulli_subtomo, target, bernoulli_mask

    def create_grid(self):
        """Create a possibly overlapping set of patches forming a grid that covers a tomogram"""
        dist_center = self.subtomo_length//2 # size from center
        centers = []
        for i, coord in enumerate(self.tomo_shape):

            n_centers = int(np.ceil(coord/self.subtomo_length))
            _centers = np.linspace(dist_center, coord-dist_center, n_centers, dtype=int)
            
            startpoints, endpoints = _centers-dist_center, _centers+dist_center
            overlap_ratio = max(endpoints[:-1]-startpoints[1::])/dist_center

            centers.append(_centers)
            
            if overlap_ratio<0:
                raise ValueError('The tomogram is not fully covered in dimension %i.' %i)
                
            # if overlap_ratio>0.5:
            #     raise ValueError('There is more than 50%% overlap between patches in dimension %i.' %i)

        zs, ys, xs = np.meshgrid(*centers, indexing='ij')
        grid = list(zip(zs.flatten(), ys.flatten(), xs.flatten()))
        
        return grid