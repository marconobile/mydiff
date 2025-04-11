import torch
import math

def center_pos(pos: torch.tensor):
    return pos - torch.mean(pos, dim=-2, keepdim=True)


# def inflate_batch_array(array, target):
#     """
#     Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
#     axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
#     """
#     target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
#     return array.view(target_shape)


# def sampled_centered_3d_noise(size, device):
#     assert len(size) == 3
#     x = torch.randn(size, device=device)
#     # This projection only works because Gaussian is rotation invariant around
#     # zero and samples are independent!
#     return center_pos(x)


def sample_noise_from_N_0_1(size, device):
    return torch.randn(size, device=device)


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))


def sigma(gamma):
    """Computes sigma given gamma."""
    return torch.sqrt(torch.sigmoid(gamma))


def alpha(gamma):
    """Computes alpha given gamma."""
    return torch.sqrt(torch.sigmoid(-gamma))




