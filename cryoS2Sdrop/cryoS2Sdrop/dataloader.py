import torch
import numpy as np
from torch.utils.data import Dataset
from tomoSegmentPipeline.utils.common import read_array
import tomopy.sim.project as proj
from tomopy.recon.algorithm import recon
from .deconvolution import tom_deconv_tomo
from scipy.stats import multivariate_normal



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
        n_shift=0,
        gt_tomo_path=None,
        **deconv_kwargs
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

        self.deconv_kwargs = {"vol":self.data.numpy(), **deconv_kwargs}
        self.use_deconv_data = self.check_deconv_kwargs(deconv_kwargs)
        if self.use_deconv_data:
            self.data = tom_deconv_tomo(**self.deconv_kwargs)
            self.data = torch.tensor(self.data)
        else:
            pass
        
        self.gt_tomo_path = gt_tomo_path
        if gt_tomo_path is not None:
            self.gt_data = torch.tensor(read_array(gt_tomo_path))
            self.gt_data = self.clip(self.gt_data)
            self.gt_data = self.standardize(self.gt_data)
        else:
            self.gt_data = None
        
        self.n_shift = n_shift
        self.tomo_shape = self.data.shape
        self.subtomo_length = subtomo_length
        self.grid = self.create_grid()
        self.transform = transform  # think how to implement this
        self.p = p
        self.Vmask_pct = Vmask_pct
        self.dropout = torch.nn.Dropout(p=p)
        self.dropoutV = torch.nn.Dropout(p=self.Vmask_pct)
        self.upsample = torch.nn.Upsample(scale_factor=volumetric_scale_factor)
        self.vol_scale_factor = volumetric_scale_factor
        self.channels = 1
        self.Vmask_probability = Vmask_probability  # otherwise use Pmask

        self.n_bernoulli_samples = n_bernoulli_samples

        self.run_init_asserts()

        return

    def check_deconv_kwargs(self, deconv_kwargs):
        if bool(deconv_kwargs):
            deconv_args = ["angpix", "defocus", "snrfalloff", "deconvstrength", "highpassnyquist"]
            for arg in deconv_args:
                if arg in self.deconv_kwargs.keys():
                    continue
                else:
                    raise KeyError('Missing required deconvolution argument: "%s"' %arg)
            use_deconv_data = True
            print('Using deconvolved data for training.')

        else:
            use_deconv_data = False
        
        return use_deconv_data

    def run_init_asserts(self):
        if self.subtomo_length % self.vol_scale_factor != 0:
            raise ValueError(
                "Length of subtomograms must be a multiple of the volumetric scale factor."
            )
        if self.subtomo_length % 32 != 0:
            raise ValueError(
                "Length of subtomograms must be a multiple of 32 to run the network."
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
        bernoulli_Vmask = self.dropoutV(torch.ones(downsampled_shape)) * (1 - self.Vmask_pct)
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
        center_z, center_y, center_x = self.shift_coords(*self.grid[index])
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

    def shift_coords(self, z, y, x):
        "Add random shift to coordinates"
        new_coords = []
        for idx, coord in enumerate([z, y, x]):
            shift_range = range(-self.n_shift, self.n_shift + 1)
            coord = coord + np.random.choice(shift_range)
            # Shift position if too close to border:
            if coord < self.subtomo_length//2:
                coord = self.subtomo_length//2
            if coord > self.tomo_shape[idx] - self.subtomo_length//2:
                coord = self.tomo_shape[idx] - self.subtomo_length//2
            new_coords.append(coord)

        return tuple(new_coords)

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
        n_shift=0,
        gt_tomo_path=None,
        input_as_target=False,
        weightedBernoulliMask_prob=0,
        **deconv_kwargs
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
            Vmask_probability, Vmask_pct, transform, n_shift, gt_tomo_path, **deconv_kwargs)
            
        self.dataF = torch.fft.rfftn(self.data)
        self.tomoF_shape = self.dataF.shape
        self.logPower_Fdata = np.log(np.abs(self.dataF)**2)
        # the spectrum still shows strong outliers
        self.logPower_Fdata = self.standardize(self.clip(self.logPower_Fdata))
        self.power_Fdata = np.exp(self.logPower_Fdata)
        
        self.highPower_mask = torch.tensor(self.power_Fdata>np.quantile(self.power_Fdata, 0.5))*1.0
        self.drawProbs = self.create_drawProbTensor()
        # here we only create one set of M bernoulli masks to be sampled from
        self.input_as_target = input_as_target
        self.weightedBernoulliMask_prob = weightedBernoulliMask_prob
        self.fourier_samples = self.create_FourierSamples()

        return

    def make_shell(self, inner_radius, delta_r, tomo_shape):
        """
        Creates a (3D) shell with given inner_radius and delta_r width centered at the middle of the array.
        
        """
        outer_radius = inner_radius + delta_r

        length = min(tomo_shape)
        mask_shape = len(tomo_shape) * [length]
        _shell_mask = np.zeros(mask_shape)

        # only do positive quadrant first
        for z in range(0, outer_radius + 1):
            for y in range(0, outer_radius + 1):
                for x in range(0, outer_radius + 1):
                

                    r = np.linalg.norm([z, y, x])

                    if r >= inner_radius and r < outer_radius:
                        zidx = z + length // 2
                        yidx = y + length // 2
                        xidx = x + length // 2

                        _shell_mask[zidx, yidx, xidx] = 1

        # first get shell for x>0
        aux = (
            np.rot90(_shell_mask, axes=(0, 1))
            + np.rot90(_shell_mask, 2, axes=(0, 1))
            + np.rot90(_shell_mask, 3, axes=(0, 1))
            + np.rot90(_shell_mask, 2, axes=(0, 2))
            + np.rot90(_shell_mask, 3, axes=(0, 2))
            + np.rot90(_shell_mask, 2, axes=(1, 2))
        )
        aux2 = _shell_mask + aux

        # finally, fill the actual shape of the tomogram with the mask
        shell_mask = np.zeros(tomo_shape)
        shell_mask[
            (tomo_shape[0] - length) // 2 : (tomo_shape[0] + length) // 2,
            (tomo_shape[1] - length) // 2 : (tomo_shape[1] + length) // 2,
            (tomo_shape[2] - length) // 2 : (tomo_shape[2] + length) // 2
        ] = aux2

        return shell_mask

    def create_Vmask(self):
        "Create volumetric blind spot random mask"
        downsampled_shape = np.array(self.tomoF_shape)// self.vol_scale_factor
        downsampled_shape = tuple(downsampled_shape)

        # we allow power correction here: not multiplying by (1-p)
        bernoulli_Vmask = self.dropoutV(torch.ones(downsampled_shape))
        bernoulli_Vmask = bernoulli_Vmask.unsqueeze(0).unsqueeze(0)
        bernoulli_Vmask = self.upsample(bernoulli_Vmask)
        extra_row = bernoulli_Vmask[...,-1].unsqueeze(-1)
        # make final shape [C, S, S, S] last row is to account for Nyquist Frequency
        bernoulli_Vmask = torch.cat([bernoulli_Vmask, extra_row], dim=-1).squeeze(0)

        if bernoulli_Vmask[0, ...].shape!=self.dataF.shape:
            raise ValueError(
                "Volumetric mask with shape %s has a different shape in the last three components as dataF with shape %s" %(str(bernoulli_Vmask.shape), str(self.dataF.shape))
                )

        return bernoulli_Vmask

    def create_drawProbTensor(self):
        z, y, x = np.mgrid[0:self.tomo_shape[0]:1, 0:self.tomo_shape[1]:1, 0:self.tomo_shape[2]:1]

        zyx = np.column_stack([z.flat, y.flat, x.flat])

        mu = np.array(self.tomo_shape)//2
        sigma = np.array(self.tomo_shape)//4
        covariance = np.diag(sigma**2)

        rv = multivariate_normal(mu, covariance)
        pdf = rv.pdf(zyx)
        pdf = pdf.reshape(self.tomo_shape)
        pdf = torch.tensor(pdf)

        # make shell correspond to the unshifted spectrum
        pdf = torch.fft.ifftshift(pdf)
        # make it correspond to only real part of spectrum
        pdf = pdf[..., 0:self.tomoF_shape[-1]]

        draw_probs = pdf/pdf.max()

        return draw_probs

    def create_weightedBernoulliMask(self):
        weighted_bernoulli_mask = torch.bernoulli(0.5*self.drawProbs+0.3*self.highPower_mask+0.05)
        weighted_bernoulli_mask = weighted_bernoulli_mask.float().unsqueeze(0)
        return weighted_bernoulli_mask

    def create_Pmask(self):
        "Create pointed blind spot random mask"
        _shape = self.tomoF_shape
        # we allow power correction here: not multiplying by (1-p)
        bernoulli_Pmask = self.dropout(torch.ones(_shape))
        bernoulli_Pmask = bernoulli_Pmask.unsqueeze(0)

        return bernoulli_Pmask

    def create_mask(self):
        if np.random.uniform() < self.weightedBernoulliMask_prob:
            mask = self.create_weightedBernoulliMask()
        else:
            mask = self.create_Pmask()

        return mask

    def create_batchFourierSamples(self, M):
        mask = torch.stack(
            [self.create_mask() for i in range(M)], axis=0,
        )
        fourier_samples = self.dataF.unsqueeze(0).repeat(
                M, 1, 1, 1, 1
            )
        fourier_samples = fourier_samples*mask
        samples = torch.fft.irfftn(fourier_samples, dim=[-3, -2, -1])

        return samples

    def create_FourierSamples(self):
        "Create a predefined set of fourier space samples that will be sampled from on each __getitem__ call"
        print('Creating Fourier samples...')
        # the factor of 2 comes from the splitting we might do afterwards to map samples to samples
        M = 2*self.n_bernoulli_samples

        samples = torch.cat([self.create_batchFourierSamples(M) for i in range(8)])
        print('Done!')

        return samples

    def __getitem__(self, index: int):
        center_z, center_y, center_x = self.shift_coords(*self.grid[index])
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

        if self.input_as_target:
            sample_idx = np.random.choice(
                range(len(self.fourier_samples)),
                self.n_bernoulli_samples,
                replace=False,
            )
            subtomo = self.fourier_samples[sample_idx][..., z_min:z_max, y_min:y_max, x_min:x_max]
            # IMPORTANT! we are mapping samples to input
            target = self.data[z_min:z_max, y_min:y_max, x_min:x_max]
            target = target.unsqueeze(0).repeat(
                self.n_bernoulli_samples, 1, 1, 1, 1
                )
        else:
            sample_idx = np.random.choice(
                range(len(self.fourier_samples)),
                2*self.n_bernoulli_samples,
                replace=False,
            )
            samples = self.fourier_samples[sample_idx][..., z_min:z_max, y_min:y_max, x_min:x_max]
            # IMPORTANT! we are mapping samples to samples
            subtomo, target = torch.split(samples, self.n_bernoulli_samples)

        if gt_subtomo is not None:
            gt_subtomo = gt_subtomo.unsqueeze(0).repeat(
                self.n_bernoulli_samples, 1, 1, 1, 1
            )

        if self.transform:
            subtomo, target, gt_subtomo = self.transform(subtomo, target, gt_subtomo)
            
        return subtomo, target, gt_subtomo


class singleCET_ProjectedDataset(Dataset):
    def __init__(
        self,
        tomo_path,
        subtomo_length,
        transform=None,
        n_shift=0,
        gt_tomo_path=None,
        predict_simRecon=False,
        use_deconv_as_target=False,
        **deconv_kwargs
    ):
        """
        Load cryoET dataset and simulate 2 independent projections for N2N denoising. All data can be optionally deconvolved.

        The dataset consists of subtomograms of shape [C, S, S, S] C (equal to 1) is the number of channels and 
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

        self.use_deconv_data = self.check_deconv_kwargs(deconv_kwargs)

        self.gt_tomo_path = gt_tomo_path
        if gt_tomo_path is not None:
            self.gt_data = torch.tensor(read_array(gt_tomo_path))
            self.gt_data = self.clip(self.gt_data)
            self.gt_data = self.standardize(self.gt_data)
        else:
            self.gt_data = None
        
        self.tomo_shape = self.data.shape
        self.subtomo_length = subtomo_length
        self.n_shift = n_shift
        self.grid = self.create_grid()
        self.transform = transform
        self.n_angles = 300
        self.shift = np.pi/np.sqrt(2) # shift by some amount that guarantees no overlap
        self.angles0 = np.linspace(0, 2*np.pi, self.n_angles)
        # shift the projection for the second reconstruction
        self.angles1 = np.linspace(0+self.shift, 2*np.pi+self.shift, self.n_angles)

        self.simRecon0 = self.make_simulated_reconstruction(self.angles0, 'fbp')
        self.simRecon0 = self.standardize(self.clip(self.simRecon0))
        self.deconv_kwargs0 = {"vol":self.simRecon0, **deconv_kwargs}
        self.simRecon0 = tom_deconv_tomo(**self.deconv_kwargs0)
        self.simRecon0 = torch.tensor(self.simRecon0)

        self.predict_simRecon = predict_simRecon
        if predict_simRecon:
            self.simRecon1 = self.make_simulated_reconstruction(self.angles1, 'fbp')
            self.simRecon1 = self.standardize(self.clip(self.simRecon1))
            # I map deconvolved to raw reconstruction because my idea is that this
            # prevents too much coupling of the noise somehow (??)
            if use_deconv_as_target:
                print('Using simRecon0 and deconvolved simRecon1 for training')
                self.deconv_kwargs1 = {"vol":self.simRecon1, **deconv_kwargs}
                self.simRecon1 = tom_deconv_tomo(**self.deconv_kwargs1)
            else:
                print('Using simRecon0 and simRecon1 for training')
            self.simRecon1 = torch.tensor(self.simRecon1)

        else:
            if use_deconv_as_target:
                print('Using simRecon0 and deconvolved data for training')
                self.deconv_kwargs = {"vol":self.data.numpy(), **deconv_kwargs}
                self.data = tom_deconv_tomo(**self.deconv_kwargs)
                self.data = torch.tensor(self.data)
            else:
                print('Using simRecon0 and data for training')

        self.use_deconv_as_target = use_deconv_as_target

        self.run_init_asserts()

        return

    def check_deconv_kwargs(self, deconv_kwargs):
        if bool(deconv_kwargs):
            deconv_args = ["angpix", "defocus", "snrfalloff", "deconvstrength", "highpassnyquist"]
            for arg in deconv_args:
                if arg in deconv_kwargs.keys():
                    continue
                else:
                    raise KeyError('Missing required deconvolution argument: "%s"' %arg)
            use_deconv_data = True

        else:
            use_deconv_data = False
        
        return use_deconv_data

    def run_init_asserts(self):
        if self.subtomo_length % 32 != 0:
            raise ValueError(
                "Length of subtomograms must be a multiple of 32 to run the network."
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


    def shift_coords(self, z, y, x):
        "Add random shift to coordinates"
        new_coords = []
        for idx, coord in enumerate([z, y, x]):
            shift_range = range(-self.n_shift, self.n_shift + 1)
            coord = coord + np.random.choice(shift_range)
            # Shift position if too close to border:
            if coord < self.subtomo_length//2:
                coord = self.subtomo_length//2
            if coord > self.tomo_shape[idx] - self.subtomo_length//2:
                coord = self.tomo_shape[idx] - self.subtomo_length//2
            new_coords.append(coord)

        return tuple(new_coords)

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

    def make_simulated_reconstruction(self, angles, algorithm):
        projection = proj.project(self.data, angles)
        reconstruction = recon(projection, angles, algorithm=algorithm)

        _shape = np.array(reconstruction.shape)
        s0 = (_shape-self.tomo_shape)//2
        s1 = _shape-s0

        reconstruction = reconstruction[s0[0]:s1[0], s0[1]:s1[1], s0[2]:s1[2]]

        return reconstruction

    def __getitem__(self, index: int):
        center_z, center_y, center_x = self.shift_coords(*self.grid[index])

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
            gt_subtomo = torch.tensor(gt_subtomo).unsqueeze(0)
        else:
            gt_subtomo = None

        subtomo = self.simRecon0[z_min:z_max, y_min:y_max, x_min:x_max]
        subtomo = torch.tensor(subtomo).unsqueeze(0)
        if self.predict_simRecon:
            target = self.simRecon1[z_min:z_max, y_min:y_max, x_min:x_max]
            target = torch.tensor(target).unsqueeze(0)
        else:
            target = self.data[z_min:z_max, y_min:y_max, x_min:x_max]
            target = torch.tensor(target).unsqueeze(0)
        
        if self.transform is not None:
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

class randomRotation3D_fourierSamples(object):
    def __init__(self, p):
        assert p >= 0 and p <= 1
        self.p = p
        
    def make3D_rotation(self, subtomo, target, gt_subtomo):
        "3D rotation in ZYX sets of images."
        # 180º rotation around Y axis
        if np.random.uniform() < self.p:
            subtomo = torch.rot90(subtomo, k=2, dims=(0, 2))
            target = torch.rot90(target, k=2, dims=(0, 2))
            gt_subtomo = torch.rot90(gt_subtomo, k=2, dims=(0, 2))
        # 180º rotation around X axis
        if np.random.uniform() < self.p:
            subtomo = torch.rot90(subtomo, k=2, dims=(0, 1))
            target = torch.rot90(target, k=2, dims=(0, 1))
            gt_subtomo = torch.rot90(gt_subtomo, k=2, dims=(0, 1))
        # rotation between 90º and 270º around Z axis
        if np.random.uniform() < self.p:
            k = int(np.random.choice([1, 2, 3]))
            subtomo = torch.rot90(subtomo, k=k, dims=(1, 2))
            target = torch.rot90(target, k=k, dims=(1, 2))
            gt_subtomo = torch.rot90(gt_subtomo, k=k, dims=(1, 2))
        
        return subtomo, target, gt_subtomo

    def __call__(self, subtomo, target, gt_subtomo):
        """
        Input are of shape [M, C, S, S, S]
        First flatten the arrays, then apply the rotations on the 4D arrays, then reshape to original shape.
        """
    
        s, t = subtomo.flatten(start_dim=0, end_dim=1), target.flatten(start_dim=0, end_dim=1)
        if gt_subtomo is not None:
            g = gt_subtomo.flatten(start_dim=0, end_dim=1)
        else:
            g = torch.zeros_like(s)
            
        subtomo_rotated, target_rotated, gt_subtomo_rotated = [], [], []

        for values in zip(s, t, g):
            a, b, c = self.make3D_rotation(*values)
            subtomo_rotated.append(a)
            target_rotated.append(b)
            gt_subtomo_rotated.append(c)

        subtomo_rotated = torch.stack(subtomo_rotated).reshape(subtomo.shape)
        target_rotated = torch.stack(target_rotated).reshape(target.shape)
        gt_subtomo_rotated = torch.stack(gt_subtomo_rotated).reshape(gt_subtomo.shape)
        
        # deal with no gt_subtomo case. Maybe not the best, since we calculate 1 too many rotations
        if (gt_subtomo_rotated==0).all():
            gt_subtomo_rotated = None
        
        return subtomo_rotated, target_rotated, gt_subtomo_rotated

    def __repr__(self):
        return repr("randomRotation3D with probability %.02f" % self.p)