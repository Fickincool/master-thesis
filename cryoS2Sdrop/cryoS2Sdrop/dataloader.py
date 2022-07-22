import torch
import numpy as np
from torch.utils.data import Dataset
from tomoSegmentPipeline.utils.common import read_array


class singleCET_dataset(Dataset):
    def __init__(
        self,
        tomo_path,
        subtomo_length,
        p,
        n_bernoulli_samples=6,
        volumetric_scale_factor=4,
        Vmask_probability=0,
        Vmask_pct=0.1,
        transform=None,
        gt_tomo_path=None
    ):
        """
        Load cryoET dataset for self2self denoising.

        The dataset consists of subtomograms of shape [M, C, S, S, S] C (equal to 1) is the number of channels and 
        S is the subtomogram side length.

        - tomo_path: tomogram path
        - subtomo_length: side length of the patches to be used for training
        - p: probability of an element to be zeroed
        - volumetric_scale_factor: times the original tomogram shape will be reduced 
        to take bernoulli point samples before upsampling into volumetric bernoulli blind spots.
        """
        self.tomo_path = tomo_path
        self.data = torch.tensor(read_array(tomo_path))
        self.data = self.clip(self.data)
        self.data = self.standardize(self.data)
        self.gt_tomo_path = gt_tomo_path
        if gt_tomo_path is not None:
            self.gt_data = torch.tensor(read_array(gt_tomo_path))
            self.gt_data = self.clip(self.gt_data)
            self.gt_data = self.standardize(self.gt_data)
        else:
            self.gt_data = None
        
        self.tomo_shape = self.data.shape
        self.subtomo_length = subtomo_length
        self.grid = self.create_grid()
        self.transform = transform  # think how to implement this
        self.p = p
        self.p0 = Vmask_pct
        self.dropout = torch.nn.Dropout(p=p)
        self.dropoutV = torch.nn.Dropout(p=self.p0)
        self.upsample = torch.nn.Upsample(scale_factor=volumetric_scale_factor)
        self.vol_scale_factor = volumetric_scale_factor
        self.channels = 1
        self.Vmask_probability = Vmask_probability  # otherwise use Pmask

        self.n_bernoulli_samples = n_bernoulli_samples

        self.run_init_asserts()

        return

    def run_init_asserts(self):
        if self.subtomo_length % self.vol_scale_factor != 0:
            raise ValueError(
                "Length of subtomograms must be a multiple of the volumetric scale factor."
            )

        return

    def standardize(self, X: torch.tensor):
        mean = X.mean()
        std = X.std()

        new_X = (X - mean) / std

        return new_X

    def clip(self, X, low=0.005, high=0.995):
        # works with tensors =)
        return np.clip(X, np.quantile(X, low), np.quantile(X, high))

    def __len__(self):
        return len(self.grid)

    def create_Vmask(self):
        "Create volumetric blind spot random mask"
        downsampled_shape = np.array(3 * [self.subtomo_length]) // self.vol_scale_factor
        downsampled_shape = tuple(downsampled_shape)

        # avoid power correction from dropout and set shape for upsampling
        bernoulli_Vmask = self.dropoutV(torch.ones(downsampled_shape)) * (1 - self.p0)
        bernoulli_Vmask = bernoulli_Vmask.unsqueeze(0).unsqueeze(0)
        # make final shape [C, S, S, S]
        bernoulli_Vmask = self.upsample(bernoulli_Vmask).squeeze(0)

        return bernoulli_Vmask

    def create_Pmask(self):
        "Create pointed blind spot random mask"
        _shape = 3 * [self.subtomo_length]
        bernoulli_Pmask = self.dropout(torch.ones(_shape)) * (1 - self.p)
        bernoulli_Pmask = bernoulli_Pmask.unsqueeze(0)

        return bernoulli_Pmask

    def create_bernoulliMask(self):
        if np.random.uniform() < self.Vmask_probability:
            # might work as an augmentation technique.
            bernoulli_mask = self.create_Vmask()
        else:
            bernoulli_mask = self.create_Pmask()

        return bernoulli_mask

    def __getitem__(self, index: int):
        center_z, center_y, center_x = self.grid[index]
        z_min, z_max = (
            center_z - self.subtomo_length // 2,
            center_z + self.subtomo_length // 2,
        )
        y_min, y_max = (
            center_y - self.subtomo_length // 2,
            center_y + self.subtomo_length // 2,
        )
        x_min, x_max = (
            center_x - self.subtomo_length // 2,
            center_x + self.subtomo_length // 2,
        )
        subtomo = self.data[z_min:z_max, y_min:y_max, x_min:x_max]
        
        if self.gt_data is not None:
            gt_subtomo = self.gt_data[z_min:z_max, y_min:y_max, x_min:x_max]
        else:
            gt_subtomo = None

        # first transform and then get samples
        if self.transform:
            subtomo, gt_subtomo = self.transform(subtomo, gt_subtomo)

        ##### One different mask per __getitem__ call
        bernoulli_mask = torch.stack(
            [self.create_bernoulliMask() for i in range(self.n_bernoulli_samples)],
            axis=0,
        )
        ##### Take samples from a pool of predefined bernoulli masks
        # bernoulli_mask_sample_idx = np.random.choice(
        #     range(len(self.bernoulli_mask_samples)),
        #     self.n_bernoulli_samples,
        #     replace=False,
        # )
        # bernoulli_mask = self.bernoulli_mask_samples[bernoulli_mask_sample_idx]
        if gt_subtomo is not None:
            gt_subtomo = gt_subtomo.unsqueeze(0).repeat(
                self.n_bernoulli_samples, 1, 1, 1, 1
            )
            
        _samples = subtomo.unsqueeze(0).repeat(
            self.n_bernoulli_samples, 1, 1, 1, 1
        )  # get n samples
        bernoulli_subtomo = bernoulli_mask * _samples  # bernoulli samples
        target = (1 - bernoulli_mask) * _samples  # complement of the bernoulli sample

        return bernoulli_subtomo, target, bernoulli_mask, gt_subtomo

    def create_grid(self):
        """Create a possibly overlapping set of patches forming a grid that covers a tomogram"""
        dist_center = self.subtomo_length // 2  # size from center
        centers = []
        for i, coord in enumerate(self.tomo_shape):

            n_centers = int(np.ceil(coord / self.subtomo_length))
            _centers = np.linspace(
                dist_center, coord - dist_center, n_centers, dtype=int
            )

            startpoints, endpoints = _centers - dist_center, _centers + dist_center
            overlap_ratio = max(endpoints[:-1] - startpoints[1::]) / dist_center

            centers.append(_centers)

            if overlap_ratio < 0:
                raise ValueError(
                    "The tomogram is not fully covered in dimension %i." % i
                )

            # if overlap_ratio>0.5:
            #     raise ValueError('There is more than 50%% overlap between patches in dimension %i.' %i)

        zs, ys, xs = np.meshgrid(*centers, indexing="ij")
        grid = list(zip(zs.flatten(), ys.flatten(), xs.flatten()))

        return grid


class singleCET_FourierDataset(singleCET_dataset):
    def __init__(
        self,
        tomo_path,
        subtomo_length,
        p,
        n_bernoulli_samples=6,
        volumetric_scale_factor=4,
        Vmask_probability=0,
        Vmask_pct=0.1,
        transform=None,
        gt_tomo_path=None
    ):
        """
        Load cryoET dataset with samples taken by Bernoulli sampling Fourier space for self2self denoising.

        The dataset consists of subtomograms of shape [M, C, S, S, S] C (equal to 1) is the number of channels and 
        S is the subtomogram side length.

        - tomo_path: tomogram path
        - subtomo_length: side length of the patches to be used for training
        - p: probability of an element to be zeroed
        - volumetric_scale_factor: times the original tomogram shape will be reduced 
        to take bernoulli point samples before upsampling into volumetric bernoulli blind spots.
        """
        singleCET_dataset.__init__(
            self, tomo_path, subtomo_length, p, n_bernoulli_samples, volumetric_scale_factor, 
            Vmask_probability, Vmask_pct, transform, gt_tomo_path)
            
        self.dataF = torch.fft.rfftn(self.data)
        self.tomoF_shape = self.dataF.shape
        # here we only create one set of M bernoulli masks to be sampled from
        self.fourier_samples = self.create_FourierSamples()

        return

    def create_Vmask(self):
        "Create volumetric blind spot random mask"
        downsampled_shape = np.array(self.tomoF_shape)// self.vol_scale_factor
        downsampled_shape = tuple(downsampled_shape)

        # we allow power correction here: not multiplying by (1-p)
        bernoulli_Vmask = self.dropoutV(torch.ones(downsampled_shape))
        bernoulli_Vmask = bernoulli_Vmask.unsqueeze(0).unsqueeze(0)
        # make final shape [C, S, S, S]
        bernoulli_Vmask = self.upsample(bernoulli_Vmask).squeeze(0)

        return bernoulli_Vmask

    def create_Pmask(self):
        "Create pointed blind spot random mask"
        _shape = self.tomoF_shape
        # we allow power correction here: not multiplying by (1-p)
        bernoulli_Pmask = self.dropout(torch.ones(_shape))
        bernoulli_Pmask = bernoulli_Pmask.unsqueeze(0)

        return bernoulli_Pmask

    def create_bernoulliMask(self):
        if np.random.uniform() < self.Vmask_probability:
            # might work as an augmentation technique.
            bernoulli_mask = self.create_Vmask()
        else:
            bernoulli_mask = self.create_Pmask()

        return bernoulli_mask

    def create_batchFourierSamples(self, M):
        bernoulli_mask = torch.stack(
            [self.create_bernoulliMask() for i in range(M)], axis=0,
        )
        fourier_samples = self.dataF.unsqueeze(0).repeat(
                M, 1, 1, 1, 1
            )
        fourier_samples = fourier_samples*bernoulli_mask
        samples = torch.fft.irfftn(fourier_samples)

        return samples

    def create_FourierSamples(self):
        "Create a predefined set of fourier space samples that will be sampled from on each __getitem__ call"
        print('Creating Fourier samples...')
        M = 2*self.n_bernoulli_samples
        samples = torch.cat([self.create_batchFourierSamples(M) for i in range(1)])
        print('Done!')

        return samples

    def __getitem__(self, index: int):
        center_z, center_y, center_x = self.grid[index]
        z_min, z_max = (
            center_z - self.subtomo_length // 2,
            center_z + self.subtomo_length // 2,
        )
        y_min, y_max = (
            center_y - self.subtomo_length // 2,
            center_y + self.subtomo_length // 2,
        )
        x_min, x_max = (
            center_x - self.subtomo_length // 2,
            center_x + self.subtomo_length // 2,
        )

        if self.gt_data is not None:
            gt_subtomo = self.gt_data[z_min:z_max, y_min:y_max, x_min:x_max]
        else:
            gt_subtomo = None

        sample_idx = np.random.choice(
            range(len(self.fourier_samples)),
            2*self.n_bernoulli_samples,
            replace=False,
        )
        samples = self.fourier_samples[sample_idx][..., z_min:z_max, y_min:y_max, x_min:x_max]
        subtomo, target = torch.split(samples, self.n_bernoulli_samples)

        if gt_subtomo is not None:
            gt_subtomo = gt_subtomo.unsqueeze(0).repeat(
                self.n_bernoulli_samples, 1, 1, 1, 1
            )

        if self.transform:
            subtomo, target, gt_subtomo = self.transform(subtomo, target, gt_subtomo)
            
        return subtomo, target, gt_subtomo

class randomRotation3D(object):
    def __init__(self, p):
        assert p >= 0 and p <= 1
        self.p = p

    def __call__(self, subtomo, gt_subtomo):
        "Input is a 3D ZYX (sub)tomogram"
        # 180º rotation around Y axis
        if np.random.uniform() < self.p:
            subtomo = torch.rot90(subtomo, k=2, dims=(0, 2))
            if gt_subtomo is not None:
                gt_subtomo = torch.rot90(gt_subtomo, k=2, dims=(0, 2))
        # 180º rotation around X axis
        if np.random.uniform() < self.p:
            subtomo = torch.rot90(subtomo, k=2, dims=(0, 1))
            if gt_subtomo is not None:
                gt_subtomo = torch.rot90(gt_subtomo, k=2, dims=(0, 1))
        # rotation between 90º and 270º around Z axis
        if np.random.uniform() < self.p:
            k = int(np.random.choice([1, 2, 3]))
            subtomo = torch.rot90(subtomo, k=k, dims=(1, 2))
            if gt_subtomo is not None:
                gt_subtomo = torch.rot90(gt_subtomo, k=k, dims=(1, 2))

        return subtomo, gt_subtomo

    def __repr__(self):
        return repr("randomRotation3D with probability %.02f" % self.p)
