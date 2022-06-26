from joblib import Parallel, delayed
import numpy as np
import torch
from torch.utils.data import Dataset


def clip_and_standardize(X: np.array, low=0.005, high=0.995, quantiles=True):

    if quantiles:
        X_clp = np.clip(X, np.quantile(X, low), np.quantile(X, high))
    else:
        X_clp = np.clip(X, low, high)

    X_stdz = (X_clp - np.mean(X_clp)) / np.std(X_clp)

    return X_stdz


def _normalize(X: np.array):
    return (X - X.min()) / (X.max() - X.min())


class destripeDataSet(Dataset):
    """
    For an input image of shape ZYX, returns an array of shape (Y, 3, Z, X) where, for a given slice Y=y:
    - (y, 0, Z, X): image ZX data
    - (y, 1, Z, X): outlier mask (based on quantiles of the weight mask)
    - (y, 2, Z, X): weight mask (normalized log power)
    """
    def __init__(self, path, normalize=True, logTransform=False):

        # TODO: add this to standalone destripe
        from tomoSegmentPipeline.utils.common import read_array

        # data is originally in ZYX form
        image_data = read_array(path)
        # we set data in YZX form
        image_data = image_data.transpose(1, 0, 2)

        # # Yu's dummies
        # pilot_file = os.path.join(PARENT_PATH, 'destripe/Data/simu-small-constant.mat')
        # image_data = io.loadmat(pilot_file)['datas'].transpose(2, 0, 1)[:,:,0:724]

        if logTransform:
            image_data = np.log10(image_data)

        if normalize:
            image_data = Parallel(n_jobs=6)(
                delayed(clip_and_standardize)(xz_plane) for xz_plane in image_data
            )
            image_data = np.array(image_data)
            # image_data = image_data - image_data.min()

        fft_img = np.array(
            [np.fft.fftshift(np.fft.fft2(xz_plane)) for xz_plane in image_data]
        )  # just run fft on the image information

        logPower_data = np.log(np.abs(fft_img) ** 2)
        logPower_data = Parallel(n_jobs=6)(
            delayed(clip_and_standardize)(xz_plane, low=0.001, high=0.999)
            for xz_plane in logPower_data
        )
        logPower_data = Parallel(n_jobs=6)(
            delayed(_normalize)(xz_plane) for xz_plane in logPower_data
        )
        logPower_data = np.array(logPower_data)

        # all image data is in the form: YZX
        # data[:, 0, :, :] = image
        # data[:, 1, :, :] = outlier mask (logPower mask)
        # data[:, 2, :, :] = weights (logPower)
        data = image_data[:, np.newaxis, :, :]

        # get outliers according to the power criterion
        outlier_mask = np.array(
            [
                (xz_plane < np.quantile(xz_plane, 0.95)).astype(int)
                for xz_plane in logPower_data
            ]
        )
        outlier_mask = outlier_mask[
            :, np.newaxis, :, :
        ]  # set a consistent shape of the data

        # get proper shape of the weight data
        weight_matrix = logPower_data[:, np.newaxis, :, :]

        # concatenate stuff
        data = np.concatenate((data, outlier_mask), 1)
        data = np.concatenate((data, weight_matrix), 1)


        self.x_data = torch.from_numpy(data).float()
        self.y_data = torch.from_numpy(np.zeros((3, 1))).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.size(0)
