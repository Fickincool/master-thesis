import torch
import numpy as np
from torch.utils.data import Dataset
from tomoSegmentPipeline.utils.common import read_array

class singleCET_dataset(Dataset):
    def __init__(self, tomo_path, subtomo_length, p, n_samples, transform=None):  
        """
        Load cryoET dataset for self2self denoising.

        The dataset consists of subtomograms of shape [n_samples, s, s, s] 
        where n_samples is the number of Bernoulli samples and s is the subtomogram side length.

        - tomo_path: tomogram path
        - subtomo_length: side length of the patches to be used for training
        - n_samples: number of independent bernoulli samples
        - p: probability of an element to be zeroed
        """
        self.tomo_path = tomo_path
        self.data = torch.tensor(read_array(tomo_path))
        self.data = self.normalize(self.data)
        self.tomo_shape = self.data.shape
        self.subtomo_length = subtomo_length
        self.grid = self.create_grid()
        self.transform = transform # think how to implement this
        self.p = p
        self.dropout = torch.nn.Dropout(p=p)
        self.n_samples = n_samples
        ####### maybe clip values, but it may alter the data distribution (?)

    def normalize(self, X:torch.tensor):
        mean = X.mean()
        std = X.std()
        normalized_data = (X - mean) / std
        return normalized_data

    def __len__(self):
        return len(self.grid)

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
        bernoulli_mask = torch.stack([self.dropout(torch.ones_like(subtomo))*(1-self.p) for i in range(self.n_samples)], axis=0)
        
        _samples = subtomo.unsqueeze(0).repeat(self.n_samples, 1, 1, 1) # get n samples
        bernoulli_subtomo = bernoulli_mask*_samples  # bernoulli samples
        target = (1-bernoulli_mask)*_samples # complement of the bernoulli sample

        return bernoulli_subtomo, target, bernoulli_mask

    def create_grid(self):
        """Create a equispaced grid for a tomogram"""
        centers = []
        for length in self.tomo_shape:
            centers.append(np.arange(self.subtomo_length//2, length - self.subtomo_length//2, self.subtomo_length))
        
        zs, ys, xs = np.meshgrid(*centers, indexing='ij')
        grid = list(zip(zs.flatten(), ys.flatten(), xs.flatten()))
        
        return grid