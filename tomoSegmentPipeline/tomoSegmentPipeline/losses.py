# ============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# ============================================================================================
# Copyright (c) 2019 - now
# Inria - Centre de Rennes Bretagne Atlantique, France
# Author: Emmanuel Moebel (serpico team); adapted by Lorenz Lamm
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# ============================================================================================

import torch


def Tversky_index(y_pred, y_true):
    # the case of α = β = 0.5 the Tversky index simplifies to be the same as the Dice coefficient
    alpha = torch.empty((1), device=("cuda" if torch.cuda.is_available() else "cpu"))
    alpha[0] = 0.5
    beta = torch.empty((1), device=("cuda" if torch.cuda.is_available() else "cpu"))
    beta[0] = 0.5
    # beta = torch.Tensor([0.5])

    # only classes 0 and 1 are taken into account for the loss
    batch_size, _, z_shp, y_shp, x_shp = y_true.shape

    # This yields a mask with the same shape as y_true[:, 0:2, :, :, :]
    mask = y_true[:, 2, :, :, :] != 1
    mask = mask.reshape(batch_size, 1, z_shp, y_shp, x_shp)
    mask = torch.stack(2 * [mask], dim=1).squeeze(2) * 1

    # we set all the coordinates labelled as 2 to zero for the loss
    y_true = y_true[:, 0:2, :, :, :] * mask
    y_pred = y_pred[:, 0:2, :, :, :] * mask

    ones = torch.ones_like(y_true)
    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true

    num = torch.sum(
        p0 * g0, dim=(0, 2, 3, 4)
    ).cuda()  # shape of inputs are (batch_size, N_class, Z, Y, X)
    den = (
        num.cuda()
        + alpha.cuda() * torch.sum(p0.cuda() * g1.cuda(), dim=(0, 2, 3, 4))
        + beta.cuda() * torch.sum(p1.cuda() * g0.cuda(), dim=(0, 2, 3, 4))
    )

    T_byClass = num / den

    return T_byClass


def Tversky_index_full(y_pred, y_true):
    # the case of α = β = 0.5 the Tversky index simplifies to be the same as the Dice coefficient
    alpha = torch.empty((1), device=("cuda" if torch.cuda.is_available() else "cpu"))
    alpha[0] = 0.5
    beta = torch.empty((1), device=("cuda" if torch.cuda.is_available() else "cpu"))
    beta[0] = 0.5
    # beta = torch.Tensor([0.5])

    ones = torch.ones_like(y_true)
    p0 = y_pred
    p1 = ones - y_pred
    g0 = y_true
    g1 = ones - y_true

    num = torch.sum(
        p0 * g0, dim=(0, 2, 3, 4)
    ).cuda()  # shape of inputs are (batch_size, N_class, Z, Y, X)
    den = (
        num.cuda()
        + alpha.cuda() * torch.sum(p0.cuda() * g1.cuda(), dim=(0, 2, 3, 4))
        + beta.cuda() * torch.sum(p1.cuda() * g0.cuda(), dim=(0, 2, 3, 4))
    )

    T_byClass = num / den

    return T_byClass


# class Tversky_loss(torch.nn.Module):
#     def __init__(self):
#         super(Tversky_loss, self).__init__()

#     def forward(self, y_true, y_pred):
#         # alpha = torch.Tensor([0.5], device=('cuda' if torch.cuda.is_available() else 'cpu)')
#         # the case of α = β = 0.5 the Tversky index simplifies to be the same as the Dice coefficient
#         alpha = torch.empty((1), device=('cuda' if torch.cuda.is_available() else 'cpu'))
#         alpha[0] = 0.5
#         beta = torch.empty((1), device=('cuda' if torch.cuda.is_available() else 'cpu'))
#         beta[0] = 0.5
#         # only classes 0 and 1 are taken into account for the loss
#         # this works, but the loss can never be zero because we have places were y_true is both not class0 and not class1
#         y_true = y_true[:, 0:2, :, :, :]
#         y_pred = y_pred[:, 0:2, :, :, :]
#         # beta = torch.Tensor([0.5])
#         ones = torch.ones_like(y_true)
#         p0 = y_pred
#         p1 = ones - y_pred
#         g0 = y_true
#         g1 = ones - y_true

#         num = torch.sum(p0 * g0, dim=(0, 2, 3, 4)) # shape of inputs are (batch_size, N_class, Z, Y, X)
#         den = num.cuda() + alpha.cuda() * torch.sum(p0.cuda() * g1.cuda(), dim=(0, 2, 3, 4)) + beta.cuda() * torch.sum(p1.cuda() * g0.cuda(), dim=(0, 2, 3, 4))

#         # Here we are getting the total tversky index for all classes that's why we return Ncl-T
#         T = torch.sum(num / den)

#         # Ncl = torch.Tensor(y_true.shape[-1])
#         Ncl = torch.empty((1), device=('cuda' if torch.cuda.is_available() else 'cpu'))
#         Ncl[0] = y_true.shape[1]
#         return Ncl - T


class Tversky_loss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        return 

    def compute_binary_T(self, y_pred, y_true):
        """
        Tversky loss only for the first two [0, 1] labels.
        Label 2 is set to 0 for both pred and true so it doesnt contribute to the loss
        """

        assert y_true.shape[1] == 3

        # only classes 0 and 1 are taken into account for the loss
        batch_size, _, z_shp, y_shp, x_shp = y_true.shape

        mask = y_true[:, 2, :, :, :] != 1
        mask = mask.reshape(batch_size, 1, z_shp, y_shp, x_shp)
        mask = torch.stack(2 * [mask], dim=1).squeeze(2) * 1

        # we set all the coordinates labelled as 2 to zero for the loss
        y_true = y_true[:, 0:2, :, :, :] * mask
        y_pred = y_pred[:, 0:2, :, :, :] * mask
        Ncl = 2

        ones = torch.ones_like(y_true)
        p0 = y_pred
        p1 = ones - y_pred
        g0 = y_true
        g1 = ones - y_true

        num = torch.sum(
            p0 * g0, dim=(0, 2, 3, 4)
        )  # shape of inputs are (batch_size, N_class, Z, Y, X)
        den = (
            num.cuda()
            + self.alpha * torch.sum(p0.cuda() * g1.cuda(), dim=(0, 2, 3, 4))
            + self.beta * torch.sum(p1.cuda() * g0.cuda(), dim=(0, 2, 3, 4))
        )

        T = num/den

        return T, Ncl

    def forward(self, y_pred, y_true):

        T, Ncl = self.compute_binary_T(y_pred, y_true)

        # Here we are getting the total tversky index for all classes that's why we return Ncl-T
        T = T.sum()

        return Ncl - T


class Tversky1_loss(Tversky_loss):
    def __init__(self, alpha=0.5, beta=0.5):
        Tversky_loss.__init__(self, alpha, beta)

    def forward(self, y_pred, y_true):

        # take only loss for class 1 (membranes) for the loss 
        T, Ncl = self.compute_binary_T(y_pred, y_true)
        T = T[1]

        return 1 - T


class Tversky_loss_full(torch.nn.Module):
    def __init__(self):
        super(Tversky_loss_full, self).__init__()

    def forward(self, y_pred, y_true):
        # alpha = torch.Tensor([0.5], device=('cuda' if torch.cuda.is_available() else 'cpu)')
        # the case of α = β = 0.5 the Tversky index simplifies to be the same as the Dice coefficient
        alpha = torch.empty(
            (1), device=("cuda" if torch.cuda.is_available() else "cpu")
        )
        alpha[0] = 0.5
        beta = torch.empty((1), device=("cuda" if torch.cuda.is_available() else "cpu"))
        beta[0] = 0.5

        ones = torch.ones_like(y_true)
        p0 = y_pred
        p1 = ones - y_pred
        g0 = y_true
        g1 = ones - y_true

        num = torch.sum(
            p0 * g0, dim=(0, 2, 3, 4)
        )  # shape of inputs are (batch_size, N_class, Z, Y, X)
        den = (
            num.cuda()
            + alpha.cuda() * torch.sum(p0.cuda() * g1.cuda(), dim=(0, 2, 3, 4))
            + beta.cuda() * torch.sum(p1.cuda() * g0.cuda(), dim=(0, 2, 3, 4))
        )

        # Here we are getting the total tversky index for all classes that's why we return Ncl-T
        T = torch.sum(num / den)

        Ncl = torch.empty((1), device=("cuda" if torch.cuda.is_available() else "cpu"))
        Ncl[0] = y_true.shape[1]

        return NCl - T
