import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def make_shell(inner_radius, delta_r, tomo_shape):
    """
    Creates a (2D) shell with given inner_radius and delta_r width centered at the middle of the array.
    
    """
    outer_radius = inner_radius + delta_r

    length = min(tomo_shape)
    mask_shape = len(tomo_shape) * [length]
    _shell_mask = np.zeros(mask_shape)

    # only do positive quadrant first
    for y in range(0, outer_radius + 1):
        for x in range(0, outer_radius + 1):

            r = np.linalg.norm([y, x])

            if r >= inner_radius and r < outer_radius:
                yidx = y + length // 2
                xidx = x + length // 2

                _shell_mask[yidx, xidx] = 1

    # first get shell for x>0
    aux = (
        np.rot90(_shell_mask, axes=(0, 1))
        + np.rot90(_shell_mask, 2, axes=(0, 1))
        + np.rot90(_shell_mask, 3, axes=(0, 1))
    )
    aux2 = _shell_mask + aux

    # finally, fill the actual shape of the tomogram with the mask
    shell_mask = np.zeros(tomo_shape)
    shell_mask[
        (tomo_shape[0] - length) // 2 : (tomo_shape[0] + length) // 2,
        (tomo_shape[1] - length) // 2 : (tomo_shape[1] + length) // 2,
    ] = aux2

    return shell_mask


def make_symmatrix(x, y):
    """Make symmetry matrix following FFT symmetries. Final shape is (y, x)"""

    if x % 2 == 1 and y % 2 == 1:
        symmatrix = (
            np.array(
                [-1 for i in range(y * x // 2)] + [0] + [1 for i in range(y * x // 2)]
            )
            .reshape(x, y)
            .T
        )
    elif x % 2 == 0 and y % 2 == 1:
        symmatrix = (
            np.array(
                [-1 for i in range(y * (x + 1) // 2)]
                + [0]
                + [1 for i in range(y * (x - 1) // 2)]
            )
            .reshape(x, y)
            .T
        )
    elif x % 2 == 0 and y % 2 == 0:
        symmatrix = (
            np.array(
                [-1 for i in range(y * (x + 1) // 2)]
                + [1 for i in range(y * (x - 1) // 2)]
            )
            .reshape(x, y)
            .T
        )
        symmatrix[y // 2, x // 2] = 0
    else:
        symmatrix = (
            np.array([-1 for i in range(y * x // 2)] + [1 for i in range(y * x // 2)])
            .reshape(x, y)
            .T
        )
        symmatrix[y // 2, x // 2] = 0

    return symmatrix


def make_N_neg_matrix(dr, neg_mask, power_mask):
    """
    Returns a pd.DataFrame of shape (N negative components, 65) with the flattened negative frequency components as the index.
    
    The columns correspond to the index value (own neighbor) and 64 randomly choosen, "uncorrupted", neighbors (NaN whenever no neighbor is present) 
    within a ring of radius 3 of each index value. 
    
    All values are given according to the flattened arrays. We assume that we use ZX slices of YZX images (missing wedge in ZX).
    
    - neg_mask: boolean array with 1 wherever the symmatrix equals -1. Shape: (Z,X)
    - power_mask: boolean array indicating low power coefficients. Shape: (Z,X)
    """
    tomo_shape = neg_mask.shape

    # global_to_neg_mapping = np.where(neg_mask.flatten()==1)[0]

    # _mapping = zip(global_to_neg_mapping, range(len(global_to_neg_mapping)))
    # global_to_neg_mapping = dict(_mapping)

    neg_neighbors = []

    def get_neighbors(inner_radius):
        # get masks
        ring_mask = make_shell(inner_radius, dr, tomo_shape)
        ring_uncorrupted_mask = (1 - power_mask) * ring_mask
        ring_neg_mask = ring_mask * neg_mask

        # get neighbors
        ring_neg_nghbrs = np.nonzero(ring_neg_mask.flatten())[0]
        ring_uncorrupted_nghbrs = np.nonzero(ring_uncorrupted_mask.flatten())[0]

        k = min(len(ring_uncorrupted_nghbrs), 64)

        # for each negative neighbor, get a random sample of size k from the uncorrupted neighbors. The first neighbor of a point is itself.
        aux = pd.DataFrame(
            [
                np.append(
                    n,
                    np.random.choice(
                        ring_uncorrupted_nghbrs[ring_uncorrupted_nghbrs != n], k
                    ),
                )
                for n in ring_neg_nghbrs
            ],
            index=ring_neg_nghbrs,
        )

        return aux

    neg_neighbors = Parallel(n_jobs=8)(
        delayed(get_neighbors)(inner_radius)
        for inner_radius in np.arange(0, min(tomo_shape) // 2 - dr, dr)
    )
    N_neg = pd.concat(neg_neighbors)

    # make a dummy dataframe with all indices corresponding to negative entries from the symmatrix
    aux = np.nonzero(neg_mask.flatten())[0]
    aux = pd.DataFrame(aux, index=aux)

    # get the final data frame with all negative flattened indices
    all_N_neg = N_neg.join(aux, how="right", rsuffix="_r")
    all_N_neg["0"] = all_N_neg["0_r"]
    all_N_neg.drop("0_r", axis=1, inplace=True)
    all_N_neg.columns = range(65)
    all_N_neg = all_N_neg.sort_index()

    return all_N_neg
