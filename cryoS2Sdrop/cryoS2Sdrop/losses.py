import torch
import torch.nn as nn

class self2self_L2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, y_wedge, y_hat):
        """
        The loss is only considered in the pixels that are masked from the beginning.
        - y_wedge: (1-bernoulli_mask)*model(bernoulli_subtomo)
        - y_hat: (1-bernoulli_mask)*subtomo
        """
        return self.loss(y_wedge, y_hat)