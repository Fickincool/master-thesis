# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (serpico team); adapted by Lorenz Lamm
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tomoSegmentPipeline.utils import core
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def transpose_to_channels_first(in_tensor):
    in_tensor = torch.transpose(in_tensor, 0, 3)
    in_tensor = torch.transpose(in_tensor, 1, 3)
    in_tensor = torch.transpose(in_tensor, 2, 3)
    return in_tensor

def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Args:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes. If `None`, this would be inferred
          as the (largest number in `y`) + 1.
        dtype: The data type expected by the input. Default: `'float32'`.
    Returns:
        A binary matrix representation of the input. The classes axis is placed
        last.

    <<<<< Copied from tensorflow >>>>>
    """
    # print('\n\n', y)
    y = torch.tensor(y, dtype=torch.int64)
    # print('\n\n', y.shape)
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
      input_shape = tuple(input_shape[:-1])
    y = y.view(-1)

    if not num_classes:
      num_classes = torch.max(y) + 1
    n = y.shape[0]
    # print(n, num_classes)
    categorical = torch.zeros((n, num_classes), dtype=torch.float32)
    # print('\n\n', categorical)

    # print(np.unique(y))
    # print('\n\n', torch.arange(n, dtype=torch.long), y)
    categorical[torch.arange(n, dtype=torch.long), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    # print(categorical.shape )
    return categorical


class tomoSegment_dataset(Dataset):
    def __init__(self, path_data, path_target, dim_in,
                 Ncl, Lrnd, augment_data):
        
        self.path_data = path_data
        self.path_target = path_target
        self.dim_in = dim_in
        self.Ncl = Ncl
        self.p_in = np.int(np.floor(self.dim_in / 2))
        self.Lrnd = Lrnd
        self.augment_data = augment_data

        self.data_list, self.target_list = core.load_dataset(self.path_data, self.path_target, '')

        self.Ncl_labels = int(max([max(np.unique(l)) for l in self.target_list]))+1
        
        self.tomo_shape = self.data_list[0].shape
        
        self.grid = self.create_grid()

    def __len__(self):
        return len(self.grid)*len(self.data_list)

    def __getitem__(self, idx):
        return self.create_batch(idx)

    def data_augmentation(self, batch_data, batch_target):

        # 180degree rotation around Y axis
        if np.random.uniform() < 0.5:
            # print('Shifting Y...')
            batch_data = torch.rot90(batch_data, k=2, dims=(0, 2))
            batch_target = torch.rot90(batch_target, k=2, dims=(0, 2))
        # 180degree rotation around X axis
        if np.random.uniform() < 0.5:
            # print('Shifting X...')
            batch_data = torch.rot90(batch_data, k=2, dims=(0, 1))
            batch_target = torch.rot90(batch_target, k=2, dims=(0, 1))
        # rotation between 90 and 270 around Z axis   
        if np.random.uniform() < 0.5:
            k = int(np.random.choice([1, 2, 3]))
            # print('Shifting Z...', k)
            batch_data = torch.rot90(batch_data, k=k, dims=(1, 2))
            batch_target = torch.rot90(batch_target, k=k, dims=(1, 2))
            
        return batch_data, batch_target


    def shift_coords(self, z, y, x, tomodim):
        # Add random shift to coordinates:
        x = x + np.random.choice(range(-self.Lrnd, self.Lrnd+1))
        y = y + np.random.choice(range(-self.Lrnd, self.Lrnd+1))
        z = z + np.random.choice(range(-self.Lrnd, self.Lrnd+1))
        
        # Shift position if too close to border:
        if (x<self.p_in) : x = self.p_in
        if (y<self.p_in) : y = self.p_in
        if (z<self.p_in) : z = self.p_in
        if (x>tomodim[2]-self.p_in): x = tomodim[2]-self.p_in
        if (y>tomodim[1]-self.p_in): y = tomodim[1]-self.p_in
        if (z>tomodim[0]-self.p_in): z = tomodim[0]-self.p_in
        
        return x, y, z
    
    
    def create_grid(self):
        
        # I got this number from inspecting a couple of the label files
        noise_margin = 15
        
        centers = []
        for c in self.tomo_shape:
            centers.append(np.arange(max(noise_margin+self.p_in//2, self.p_in),
                                min(c-noise_margin-self.p_in//2 + 1, c-self.p_in+1),
                                self.p_in))
        
        zs, ys, xs = np.meshgrid(*centers, indexing='ij')
        grid = list(zip(zs.flatten(), ys.flatten(), xs.flatten()))
        
        return grid

    def create_batch(self, index):
        """
        Generates batches for training and validation. In this version, the whole dataset has already been loaded into
        memory, and batch is sampled from there.
        INPUTS:
          index: 
        OUTPUT:
          batch_data: numpy array [batch_idx, channel, z, y, x] in our case only 1 channel
          batch_target: numpy array [batch_idx, class_idx, z, y, x] is one-hot encoded
        """
        
        batch_data = np.zeros((self.dim_in, self.dim_in, self.dim_in, 1))
        
        inputID = index//len(self.grid)

        sample_data = self.data_list[inputID]
        sample_target = self.target_list[inputID]

        # Get patch position:
        z, y, x = self.grid[index%len(self.grid)]
        z, y, x = self.shift_coords(z, y, x, sample_data.shape)

        # Get patch:
        patch_data = sample_data[z - self.p_in:z + self.p_in, y - self.p_in:y + self.p_in, x - self.p_in:x + self.p_in]
        patch_target = sample_target[z - self.p_in:z + self.p_in, y - self.p_in:y + self.p_in, x - self.p_in:x + self.p_in]

        # Process the patches in order to be used by network:
        # print(patch_data)
        patch_data = (patch_data - torch.mean(patch_data)) / torch.std(patch_data)  # normalize
        patch_target_onehot = to_categorical(patch_target, self.Ncl_labels)

        # Store into batch array:
        batch_data[:, :, :, 0] = patch_data
        batch_target = patch_target_onehot # shape is Z, Y, X, Ncl

        batch_data = torch.tensor(batch_data)
        batch_target = torch.tensor(batch_target)
        
        ### Disable data augmentation to run overfitting to one sample
        ############## Data augmentation #############################
        if self.augment_data:
            batch_data, batch_target = self.data_augmentation(batch_data, batch_target)

        batch_data = transpose_to_channels_first(batch_data).to(torch.float)
        batch_target = transpose_to_channels_first(batch_target).to(torch.float)

        return batch_data, batch_target


class tomoSegment_dummyDataset(Dataset):

    def __init__(self, dim_in, Ncl, 
     path_data=['/home/haicu/jeronimo.carvajal/Thesis/data/nnUnet/Task143_cryoET7/imagesTr/tomo02_patch000_0000.nii.gz'],
     path_target=['/home/haicu/jeronimo.carvajal/Thesis/data/nnUnet/Task143_cryoET7/labelsTr/tomo02_patch000.nii.gz']):
        
        self.path_data = path_data
        self.path_target = path_target
        self.dim_in = dim_in
        self.Ncl = Ncl
        self.p_in = np.int(np.floor(self.dim_in / 2))

        self.data_list, self.target_list = core.load_dataset(self.path_data, self.path_target, '')

        self.Ncl_labels = int(max([max(np.unique(l)) for l in self.target_list]))+1
        
        self.tomo_shape = self.data_list[0].shape
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.create_dummyBatch(idx)

    def create_dummyBatch(self, index):
        """
        Generates batch centered in the tomogram with the given index

        INPUTS:
          index: 
        OUTPUT:
          batch_data: numpy array [batch_idx, channel, z, y, x] in our case only 1 channel
          batch_target: numpy array [batch_idx, class_idx, z, y, x] is one-hot encoded
        """
        
        batch_data = np.zeros((self.dim_in, self.dim_in, self.dim_in, 1))
        
        inputID = index

        sample_data = self.data_list[inputID]
        sample_target = self.target_list[inputID]

        # Get patch position:
        z, y, x = np.array(self.tomo_shape)//2

        # Get patch:
        patch_data = sample_data[z - self.p_in:z + self.p_in, y - self.p_in:y + self.p_in, x - self.p_in:x + self.p_in]
        patch_target = sample_target[z - self.p_in:z + self.p_in, y - self.p_in:y + self.p_in, x - self.p_in:x + self.p_in]

        # Process the patches in order to be used by network:
        # print(patch_data)
        patch_data = (patch_data - torch.mean(patch_data)) / torch.std(patch_data)  # normalize
        patch_target_onehot = to_categorical(patch_target, self.Ncl_labels)

        # Store into batch array:
        batch_data[:, :, :, 0] = patch_data
        batch_target = patch_target_onehot # shape is Z, Y, X, Ncl

        batch_data = transpose_to_channels_first(torch.tensor(batch_data)).to(torch.float)
        batch_target = transpose_to_channels_first(torch.tensor(batch_target)).to(torch.float)

        return batch_data, batch_target

class reconstructionTask_dataset(Dataset):
    """
    Similar to the deep finder data set, but for an image inpainting reconstruction task.
    
    In this case the target image is the original 3D grayscale image and the inputs are their randomly masked versions.
    """
    def __init__(self, flag_direct_read, path_data, path_target, objlist, dim_in, Lrnd, h5_dset_name, mask_pct):
        self.flag_direct_read = flag_direct_read
        self.path_data = path_data
        self.path_target = path_target
        self.objlist = objlist
        self.dim_in = dim_in
        self.p_in = np.int(np.floor(self.dim_in / 2))
        self.Lrnd = Lrnd
        
        assert mask_pct <= 0.5
        assert mask_pct >= 0
        
        self.mask_pct = mask_pct
        
        if not self.flag_direct_read:
            self.data_list, self.target_list = core.load_dataset(self.path_data, self.path_target, h5_dset_name)

    def __len__(self):
        return len(self.objlist)

    def __getitem__(self, idx):
        if self.flag_direct_read:
            return self.generate_batch_direct_read(idx)
        else:
            return self.generate_batch_from_array(idx)

    def createRandomSquare(self, mask):
        """Create a random square within an existing mask."""

        z, y, x = mask.shape[1::]

        # Patches need to be bigger than 12**3
        assert all(np.array([z, y, x])>12)

        z_c_bounds = [3, z-3]
        y_c_bounds = [3, y-3]
        x_c_bounds = [3, x-3]

        center = [np.random.randint(z_c_bounds[0], z_c_bounds[1]), np.random.randint(y_c_bounds[0], y_c_bounds[1]),
                np.random.randint(x_c_bounds[0], x_c_bounds[1])]

        z_c, y_c, x_c = center

        ############### number of pixels for each square is between bounds
        z_pixel_bounds = min([z_c, z-z_c])
        n_pixel_z = np.random.randint(2, z_pixel_bounds)

        y_pixel_bounds = min([y_c, y-y_c])
        n_pixel_y = np.random.randint(2, y_pixel_bounds)

        x_pixel_bounds = min([x_c, x-x_c])
        n_pixel_x = np.random.randint(2, x_pixel_bounds)
        
        ###############

        mask[0, z_c-n_pixel_z:z_c+n_pixel_z, y_c-n_pixel_y:y_c+n_pixel_y, x_c-n_pixel_x:x_c+n_pixel_x] = 0
        
        return mask

    def data_augmentation(self, batch_data):
        # 180degree rotation around Y axis
        if np.random.uniform() < 0.5:
            # print('Shifting Y...')
            batch_data = torch.rot90(batch_data, k=2, dims=(0, 2))
        # 180degree rotation around X axis
        if np.random.uniform() < 0.5:
            # print('Shifting X...')
            batch_data = torch.rot90(batch_data, k=2, dims=(0, 1))
        # rotation between 90 and 270 around Z axis   
        if np.random.uniform() < 0.5:
            k = int(np.random.choice([1, 2, 3]))
            # print('Shifting Z...', k)
            batch_data = torch.rot90(batch_data, k=k, dims=(1, 2))
        
        return batch_data

    def createRandomSquareMask(self, img, max_mask_pct=0.3):
        """
        Incrementally create a random square mask where at most max_mask_pct of the pixels are missing (approximately).
        
        Shape of image is (C, Z, Y, X).
        """
        ## Prepare masking matrix
        mask = np.full(img.shape, 255) ## White background
        total_volume = img.shape[1]*img.shape[2]*img.shape[3]
        masked_volume = 0
        
        for _ in range(np.random.randint(5, 20)):
            if masked_volume > total_volume*max_mask_pct:
                break
            else:
                mask = self.createRandomSquare(mask)
                masked_volume = (mask[0]==0).sum()
        
        # print("Total masked percentage: %f" %(100*masked_volume/total_volume))
            
        return torch.tensor(mask)


    def generate_batch_from_array(self, index):

        batch_data = torch.zeros((self.dim_in, self.dim_in, self.dim_in, 1))
        batch_target = torch.zeros((self.dim_in, self.dim_in, self.dim_in, 1))
        
        tomoID = int(self.objlist[index]['tomo_idx'])

        tomodim = self.data_list[tomoID].shape

        sample_data = self.data_list[tomoID]

        # Get patch position:
        x, y, z = core.get_patch_position(tomodim, self.p_in, self.objlist[index], self.Lrnd)

        # Get patch:
        patch_data = sample_data[z - self.p_in:z + self.p_in, y - self.p_in:y + self.p_in, x - self.p_in:x + self.p_in]

        # Process the patches in order to be used by network:
        patch_data = (patch_data - torch.mean(patch_data)) / torch.std(patch_data)  # normalize

        # Store into batch array:
        batch_data[:, :, :, 0] = patch_data
        
        ### Disable data augmentation to run overfitting to one sample
        # Data augmentation (180degree rotation around tilt axis):
        batch_data = self.data_augmentation(batch_data)
        
        batch_data = transpose_to_channels_first(batch_data)        
        batch_target = torch.clone(batch_data)

        # enabling the random seed yields always the same random mask
        # i decided to not use a random seed so that the mask is always different
        # np.random.seed(888)
        mask = self.createRandomSquareMask(batch_data, self.mask_pct)
        batch_data[mask==0] = 0
        
        return batch_data, batch_target

