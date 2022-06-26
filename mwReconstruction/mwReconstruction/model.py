import torch.nn as nn
import torch
import torch.fft as fft
import numpy as np
import scipy.io as io
import torch.nn.functional as F
from torch.autograd import Function
from mwReconstruction.masking import make_N_neg_matrix, make_symmatrix


class ConvDic(nn.Module):
    "Linear layer for complex numbers"

    def __init__(self, in_channels, out_channels, bias=True):
        super(ConvDic, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels, bias=bias).to(torch.cfloat)

    def forward(self, x):
        return self.conv1(x)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(GraphAttentionLayer, self).__init__()
        self.ScaleFactor = 1
        self.relu = nn.ReLU()
        self.conv1 = nn.Linear(in_channels, self.ScaleFactor, bias=True).to(
            torch.cfloat
        )

    def forward(self, Wh_neg, N_neg, k_neighbors):
        tmp = self.conv1(Wh_neg)
        Wh1_neg = tmp[N_neg[0, :], :]
        Wh2_neg = tmp[N_neg[1:, :].reshape(-1), :].reshape(
            k_neighbors, -1, self.ScaleFactor
        )
        attention_neg = self.relu(
            torch.einsum("nc,knc->knc", (Wh1_neg.real, Wh2_neg.real))
            + torch.einsum("nc,knc->knc", (Wh1_neg.imag, Wh2_neg.imag))
        ).squeeze()
        # attention_neg = self.relu(torch.einsum('nc,knc->knc', (Wh1_neg, Wh2_neg)).abs().sum(2))
        # attention_neg = self.relu(torch.einsum('nc,knc->knc', (Wh1_neg.real, Wh2_neg.real))+torch.einsum('nc,knc->knc', (Wh1_neg.imag, Wh2_neg.imag))).sum(2)/math.sqrt(self.ScaleFactor)
        return torch.einsum(
            "kn,n->kn", (attention_neg, torch.reciprocal(attention_neg.sum(0) + 1e-3))
        )


class GCN(nn.Module):
    "Graph Convolutional Network with attention"

    def __init__(self, in_channels):
        super(GCN, self).__init__()
        self.inc = in_channels
        self.conv1 = ConvDic(1, self.inc)
        self.conv2 = ConvDic(self.inc, self.inc)
        self.conv3 = ConvDic(self.inc, 1)
        self.att = GraphAttentionLayer(self.inc)
        self.relu = nn.ELU()

    def forward(
        self,
        x_neg,
        hier_mask_neg,
        N_neg,
        mask_ind_neg,
        weight_matrix_neg,
        mask_ind,
        m,
        n,
        aver,
        k_neighbors,
    ):
        """
        Inputs
        -----------------------------
        - x_neg: Fourier coefficients corresponding to the negative part of the symmetry
        - hier_mask_neg: outlier mask for x_neg
        - N_neg: list of neighbors of the flattened array x_neg, shape: (N neighbors, )
        - mask_ind_neg: binary mask corresponding to which frequencies in the (2D) Fourier space correspond to the negative part of the symmetry
        - weight_matrix_neg:
        - mask_ind: 
        - m: 
        - n: 
        - aver: average value of the image, corresponds to the central component of the power spectrum (??)
        - k_neighbors:

        Logic:
        1. Perform nonlinear transformation of x_neg separately for real and imaginary parts
        2. Second linear transformation
        3. Run attention layer
        4. Select neighbors (?)
        5. Reduce
        6. Concatenate
        """
        x_neg = self.conv1(x_neg)
        x_neg = self.relu(x_neg.real) + 1j * self.relu(x_neg.imag)
        x_neg = self.conv2(x_neg)
        L_neg = self.att(Wh_neg=x_neg, N_neg=N_neg, k_neighbors=k_neighbors)

        mystery_array0 = x_neg[N_neg[1:, :].reshape(-1), :].reshape(
            k_neighbors, -1, self.inc
        )
        xc_neg = torch.einsum("knc,kn->knc", (mystery_array0, L_neg,),).sum(0)

        mystery_array1 = (
            torch.masked_select(x_neg.permute(1, 0), hier_mask_neg == 1)
            .reshape(self.inc, -1)
            .permute(1, 0)
        )
        x_neg = torch.cat((xc_neg, mystery_array1,), 0,).index_select(0, mask_ind_neg)

        x_neg = self.conv3(x_neg)
        x_out = (
            torch.cat((x_neg, aver, torch.flip(torch.conj(x_neg), [0])), 0)
            .index_select(0, mask_ind)
            .reshape(m, n)
        )

        return x_out


class MGNNds(nn.Module):
    def __init__(self, xc, weight_matrix, outlier_mask):
        super(MGNNds, self).__init__()
        self.xc = xc
        self.weight_matrix = weight_matrix
        self.outlier_mask = outlier_mask

        _shape = xc.shape

        symmatrix = make_symmatrix(_shape[1], _shape[0])  # this is an np.array

        # make masks on initialization (TODO: write matrices to files and just read, maybe much faster. Or run in parallel)
        aux_neg = make_N_neg_matrix(
            dr=3,
            neg_mask=(symmatrix == -1).astype(int),
            power_mask=self.outlier_mask.cpu().numpy(),
        ).to_numpy()
        self.N_neg = torch.from_numpy(aux_neg.transpose(1, 0)).long().cuda()
        self.symmask = torch.from_numpy(symmatrix).float().cuda()

        self.k_neighbors = self.N_neg.size(0) - 1
        self.MGNN1 = GCN(32)
        self.conv1 = ConvDic(1, 32)

    def forward(self, xc, weight_matrix, outlier_mask):
        xcf = fft.fftshift(fft.fft2(xc))
        m, n = xcf.size()
        x0_neg = torch.masked_select(xcf, self.symmask == -1).unsqueeze(1)
        hier_mask = ((weight_matrix > 0.2) | (outlier_mask == 1)).float()
        hier_mask_neg = torch.masked_select(hier_mask, self.symmask == -1)
        weight_matrix_neg = torch.masked_select(weight_matrix, self.symmask == -1)

        _, mask_ind = torch.sort(
            torch.cat(
                [
                    torch.where(self.symmask.reshape(-1) == index)[0]
                    for index in [-1, 0, 1]
                ]
            )
        )
        _, mask_ind_neg = torch.sort(
            torch.cat(
                [
                    torch.where(hier_mask_neg.reshape(-1) == index)[0]
                    for index in range(2)
                ]
            )
        )
        N_neg = self.N_neg[:, hier_mask_neg == 0]

        x = self.MGNN1(
            x_neg=x0_neg,
            hier_mask_neg=hier_mask_neg,
            N_neg=N_neg,
            mask_ind_neg=mask_ind_neg,
            weight_matrix_neg=weight_matrix_neg,
            mask_ind=mask_ind,
            m=m,
            n=n,
            aver=xc.sum().reshape(1, 1),
            k_neighbors=self.k_neighbors,
        )
        recon = fft.ifft2(fft.ifftshift((x))).real
        stripe = xc - recon

        stripe = stripe - stripe.max()
        return recon, x, xc - stripe, stripe
