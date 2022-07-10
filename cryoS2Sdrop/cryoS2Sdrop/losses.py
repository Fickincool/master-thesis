import torch
import torch.nn as nn

class self2self_L2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_wedge, y_hat):
        """
        Tensors of shape: [B, C, S, S, S]
        The loss is only considered in the pixels that are masked from the beginning.
        - y_wedge: (1-bernoulli_mask)*model(bernoulli_subtomo)
        - y_hat: (1-bernoulli_mask)*subtomo

        The loss is the L2 norm across the image, then mean across the batch. The mean across the batch helps to deal with "incomplete"
        batches, which are usually the last ones.
        """
        return torch.linalg.vector_norm(y_wedge-y_hat, ord=2, dim=(-4, -3, -2, -1)).mean(0)