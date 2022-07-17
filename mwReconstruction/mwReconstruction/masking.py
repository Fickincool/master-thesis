from multiprocessing.sharedctypes import Value
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from operator import itemgetter


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


def make_N_neg_matrix(dr, neg_mask, power_mask, n_neighbors=64):
    """
    Returns a pd.DataFrame of shape (N negative components, 1+n_neighbors) with the set of randomly chosen uncorrupted neighbors of 
    each negative frequency component within a ring of radius dr.
    
    The values are indexed based on the flattened array of the negative part of the frequency spectrum for consistency with the GCN.

    Note: there are 2 reasons why an index might only be its own neighbor
    # 1) It is noncorrupted
    # 2) It is corrupted but has no uncorrupted neighbors

    Inputs
    ---------------------------------
    - dr: radius of the ring
    - neg_mask: boolean array with 1 wherever the symmatrix equals -1. Shape: (Z,X)
    - power_mask: boolean array indicating low power coefficients. Shape: (Z,X)
    """
    tomo_shape = neg_mask.shape

    global_to_neg_mapping = np.where(neg_mask.flatten() == 1)[0]
    _mapping = zip(global_to_neg_mapping, range(len(global_to_neg_mapping)))
    global_to_neg_mapping = dict(_mapping)

    neg_neighbors = []

    def get_neighbors(inner_radius):
        "Get uncorrupted neighbors within a ring"
        ################# get masks
        ring_mask = make_shell(inner_radius, dr, tomo_shape)
        ring_neg_mask = ring_mask * neg_mask
        # we only sample neighbors from the uncorrupted, negative part of the spectrum
        ring_uncorrupted_mask = (1 - power_mask) * ring_neg_mask
        ring_corrupted_mask = power_mask * ring_neg_mask

        ################# get neighbors
        # these sets are still based on the full image mapping
        ring_neg_nghbrs = np.nonzero(ring_neg_mask.flatten())[0]
        ring_uncorrupted_nghbrs = np.nonzero(ring_uncorrupted_mask.flatten())[0]
        # ring_corrupted_nghbrs = np.nonzero(ring_corrupted_mask.flatten())[0]

        k = min(len(ring_uncorrupted_nghbrs) - 1, n_neighbors)
        k = max(0, k)  # it might happen that we have no uncorrupted neighbors

        def retrieve_mapped_random_neighbors(n, k):
            "Map flat global indices to flat negative indices."
            neighbor_pool = ring_uncorrupted_nghbrs[ring_uncorrupted_nghbrs != n]
            if k > 0:
                neighbor_global_indices = np.random.choice(
                    neighbor_pool, k, replace=False
                )
                own_neighbor = global_to_neg_mapping[n]
                neighbors = np.append(
                    own_neighbor,
                    itemgetter(*neighbor_global_indices)(global_to_neg_mapping),
                )
            else:
                neighbors = global_to_neg_mapping[n]
            return neighbors

        # for each corrupted negative neighbor, get a random sample of size k from the uncorrupted neighbors.
        # The first neighbor of a point is itself.
        # Finally map global indices to negative indices in a way that is consistent with the GCN logic
        aux = pd.DataFrame(
            [retrieve_mapped_random_neighbors(n, k) for n in ring_neg_nghbrs],
            index=ring_neg_nghbrs,
        )

        return aux

    neg_neighbors = Parallel(n_jobs=12)(
        delayed(get_neighbors)(inner_radius)
        for inner_radius in np.arange(0, min(tomo_shape) // 2 - dr, dr)
    )
    N_neg = pd.concat(neg_neighbors)

    # make a dummy dataframe with all indices corresponding to negative entries from the symmatrix
    aux = len(np.nonzero(neg_mask.flatten())[0])
    aux = pd.DataFrame(range(aux), index=range(aux))

    # get the final data frame with all negative flattened indices
    N_neg = N_neg.join(aux, on=0, how="right", rsuffix="_r")
    N_neg["0"] = N_neg["0_r"]
    N_neg.index = N_neg["key_0"].values

    N_neg.drop(["0_r", "key_0"], axis=1, inplace=True)

    N_neg.columns = range(n_neighbors + 1)
    N_neg = N_neg.sort_index()

    return N_neg
