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
        return torch.linalg.vector_norm(
            y_wedge - y_hat, ord=2, dim=(-4, -3, -2, -1)
        ).mean(0)


def total_variation3D(img: torch.Tensor) -> torch.Tensor:
    r"""Function that computes (Anisotropic) Total Variation according to [1].

    Args:
        img: the input image with shape :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`.

    Return:
         a scalar with the computer loss.

    Examples:
        >>> total_variation(torch.ones(3, 4, 4))
        tensor(0.)

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       total_variation_denoising.html>`__.

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")

    if len(img.shape) < 4 or len(img.shape) > 5:
        raise ValueError(f"Expected input tensor to be of ndim 4 or 5, but got {len(img.shape)}.")

    pixel_dif1 = img[..., 1:, :, :] - img[..., :-1, :, :]
    pixel_dif2 = img[..., :, 1:, :] - img[..., :, :-1, :]
    pixel_dif3 = img[..., :, :, 1:] - img[..., :, :, :-1]

    reduce_axes = (-4, -3, -2, -1)

    res = 0
    for pixel_dif in [pixel_dif1, pixel_dif2, pixel_dif3]:
        res += pixel_dif.abs().sum(dim=reduce_axes)

    return res

class TotalVariation(torch.nn.Module):
    r"""Compute the Total Variation according to [1].

    Shape:
        - Input: :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        - Output: :math:`(N,)` or scalar.

    Examples:
        >>> tv = TotalVariation()
        >>> output = tv(torch.ones((2, 3, 4, 4), requires_grad=True))
        >>> output.data
        tensor([0., 0.])
        >>> output.sum().backward()  # grad can be implicitly created only for scalar outputs

    Reference:
        [1] https://en.wikipedia.org/wiki/Total_variation
    """
    def __init__(self):
        super().__init__()

    def forward(self, img) -> torch.Tensor:
        return total_variation3D(img)

class self2selfLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = self2self_L2Loss()
        self.total_variation = TotalVariation()

    def forward(self, y_wedge, y_hat):
        """
        Tensors of shape: [B, C, S, S, S]
        The loss is only considered in the pixels that are masked from the beginning.
        - y_wedge: (1-bernoulli_mask)*model(bernoulli_subtomo)
        - y_hat: (1-bernoulli_mask)*subtomo
        """
        return self.l2(y_wedge, y_hat) + 1e-4*self.total_variation(y_wedge).mean(0)