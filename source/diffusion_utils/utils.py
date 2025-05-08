import torch
import math
from typing import Tuple, Optional

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


def compute_min_max_distance(
    points: torch.Tensor
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Computes the minimum and maximum Euclidean distance among all pairs
    of distinct points in a torch tensor of shape (N, 3).

    Args:
        points: A torch tensor of shape (N, 3) representing N points
                in 3D space.

    Returns:
        A tuple containing:
        - min_dist: The minimum distance found (a scalar tensor), or None if N < 2.
        - max_dist: The maximum distance found (a scalar tensor), or None if N < 2.

    Raises:
        ValueError: If the input tensor is not of shape (N, 3).
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input tensor must have shape (N, 3)")

    N = points.shape[0]

    if N < 2:
        # Cannot compute distance between pairs if less than 2 points
        return None, None

    # Calculate pairwise Euclidean distances using torch.cdist
    # dists[i, j] is the distance between points[i] and points[j]
    # The resulting tensor 'dists' will have shape (N, N)
    dists = torch.cdist(points, points, p=2)

    # We need to exclude the distance of a point to itself (diagonal, which is 0).
    # Create a boolean mask for the diagonal elements.
    mask = torch.eye(N, dtype=torch.bool)

    # Get the distances for all pairs of *distinct* points by using the mask.
    # This flattens the matrix and removes the diagonal elements.
    pairwise_distances = dists[~mask]

    # The minimum distance among distinct points.
    # This will be 0 if any two points are identical.
    min_dist = torch.min(pairwise_distances)

    # The maximum distance among all distinct pairs.
    max_dist = torch.max(pairwise_distances)

    return min_dist, max_dist