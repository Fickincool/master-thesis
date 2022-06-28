import torch.nn as nn
import torch
import torch.fft as fft
import numpy as np
import scipy.io as io
import torch.nn.functional as F
from torch.autograd import Function
from mwReconstruction.masking import make_N_neg_matrix, make_symmatrix
import pandas as pd


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
        super().__init__()
        self.inc = in_channels
        self.conv1 = ConvDic(1, self.inc)
        self.conv2 = ConvDic(self.inc, self.inc)
        self.conv3 = ConvDic(self.inc, 1)
        self.att = GraphAttentionLayer(self.inc)
        self.relu = nn.ELU()

    def neg2full_spectrum(self, x_neg, m, n, aver):

        r = m+1 if m%2==0 else m
        s = n+1 if n%2==0 else n

        d = x_neg.device

        aux_symmatrix = make_symmatrix(s, r)
        aux_symmask = torch.from_numpy(aux_symmatrix).float().to(d)
        
        x_neg = x_neg.repeat(2)[0:(aux_symmask==-1).sum()]

        new_img = (aux_symmask).type(torch.complex128).to(d)
        new_img[torch.where(aux_symmask==-1)] = x_neg.type(torch.complex128)

        new_img = torch.flip(new_img, [0, 1])

        new_img[torch.where(new_img==1)] = torch.conj(x_neg).type(torch.complex128)
        new_img = torch.flip(new_img, [1, 0])

        new_img[r//2, s//2] = aver

        return new_img[0:m, 0:n]


    def forward( self, x_neg, N_neg, m, n, aver):
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
        """

        x_neg = self.conv1(x_neg)
        x_neg = self.relu(x_neg.real) + 1j * self.relu(x_neg.imag)
        x_neg = self.conv2(x_neg)
        # L_neg = self.att(Wh_neg=x_neg, N_neg=N_neg, k_neighbors=k_neighbors)

        # update message: sum all neighbors for each index
        message_tensor = torch.cat((x_neg, torch.zeros(1, self.inc).to(x_neg.device)))[N_neg]
        x_neg = message_tensor.sum(1)

        x_neg = self.conv3(x_neg)

        x_out = self.neg2full_spectrum(x_neg.squeeze(), m, n, aver)

        return x_out


class MGNNds(nn.Module):
    def __init__(self, xc, weight_matrix, outlier_mask):
        super().__init__()
        self.xc = xc
        self.weight_matrix = weight_matrix
        self.outlier_mask = outlier_mask

        _shape = xc.shape
        d = xc.device

        symmatrix = make_symmatrix(_shape[1], _shape[0])  # this is an np.array
        neg_mask = (symmatrix == -1).astype(int)

        # make masks on initialization (TODO: Maybe precalculate.)
        N_neg = make_N_neg_matrix(
            dr=3,
            neg_mask=neg_mask,
            power_mask=self.outlier_mask.cpu().numpy(),
            n_neighbors=64
        ).fillna(-1).to_numpy()

        self.N_neg = torch.from_numpy(N_neg).long().to(d)
        self.symmask = torch.from_numpy(symmatrix).float().to(d)

        self.k_neighbors = self.N_neg.size(1) - 1
        self.MGNN1 = GCN(32)
        self.conv1 = ConvDic(1, 32)

    def forward(self, xc, weight_matrix, outlier_mask):

        xcf = fft.fftshift(fft.fft2(xc))
        m, n = xcf.size()
        x0_neg = torch.masked_select(xcf, self.symmask == -1).unsqueeze(1)
        hier_mask_neg = torch.masked_select(outlier_mask, self.symmask == -1)
        weight_matrix_neg = torch.masked_select(weight_matrix, self.symmask == -1)

        x = self.MGNN1(x0_neg, self.N_neg, m, n, self.xc.sum())

        recon = fft.ifft2(fft.ifftshift((x))).real
        stripe = xc - recon

        stripe = stripe - stripe.max()

        return recon, x, xc - stripe, stripe
