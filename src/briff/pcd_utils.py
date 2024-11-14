from __future__ import annotations
import torch
from torch import Tensor


# _________________________________________________ Center ________________________________________________ #

def center(pointclouds: Tensor) -> Tensor:
    """ Center each element in a batch of point clouds (substract the mean over the second dim).

    Args:
        pointclouds (Tensor): Batch of point clouds. Batch `(B, N, *)`.

    Returns:
        Batch of centered point clouds, and optionally batch of computed means `(B, *)`.
    """
    means = pointclouds.mean(dim=1, keepdim=True)
    centered_pointclouds = pointclouds - means
    return centered_pointclouds


# _________________________________________________ Scale _________________________________________________ #

def scale(pointclouds: Tensor, values: Tensor) -> Tensor:
    """ Apply a unique scaling factor to each point cloud in the provided batch.

    Args:
        pointclouds (Tensor): Batch of point clouds `(B, N, *)`.
        values (Tensor): Batch of scalar scaling values `(B,)`.

    Returns:
        Batch of scaled point clouds `(B, N, *)`.
    """
    return pointclouds * values[:, None, None].to(pointclouds.device)


# ___________________________________________ Pairwise Max Norm ___________________________________________ #

def pairwise_max_norm(pointclouds1: Tensor, pointclouds2: Tensor) -> tuple[Tensor, Tensor]:
    """ Scale each pair of elements of the two batches by their maximal norm.

    !!! Warning
        `pointclouds1` and `pointclouds2` MUST have the same length.

    Args:
        pointclouds1 (Tensor): Batch of point clouds `(B, N, *)`.
        pointclouds2 (Tensor): Batch of point clouds `(B, M, *)`.

    Returns:
        The two batches of normalized point clouds.
    """
    assert len(pointclouds1) == len(pointclouds2), 'Inputs must be of same length.'
    pointclouds1_c = center(pointclouds1)
    pointclouds2_c = center(pointclouds2)
    norms1 = pointclouds1_c.norm(dim=2).amax(dim=1)  # B
    norms2 = pointclouds2_c.norm(dim=2).amax(dim=1)  # B
    max_norms = torch.stack((norms1, norms2)).amax(dim=0)  # (2, B) -> B
    pointclouds1_n = scale(pointclouds1, 1 / max_norms)
    pointclouds2_n = scale(pointclouds2, 1 / max_norms)
    return pointclouds1_n, pointclouds2_n


# _______________________________________________ Max Norm ________________________________________________ #

def max_norm(pointclouds: Tensor) -> Tensor:
    """ Apply the same scaling factor to all the point clouds in the batch, so that the biggest has a norm
        of 1.

    Args:
        pointclouds (Tensor): Batch of point clouds `(B, N, *)`.

    Returns:
        Batch of normalized point clouds.
    """
    pointclouds_c = center(pointclouds)
    max_norm = pointclouds_c.norm(dim=2).amax(dim=1).max()
    pointclouds_n = scale(pointclouds, 1 / max_norm.repeat(len(pointclouds)))
    return pointclouds_n


# ___________________________________ Flatten, Unflatten, Stack, Expand ___________________________________ #

def flatten(x: Tensor) -> Tensor:
    return x.flatten(start_dim=0, end_dim=1)


def unflatten(x: Tensor, sizes: tuple[int, int]) -> Tensor:
    return x.unflatten(0, sizes)


def stack_and_flatten(pointclouds: tuple[Tensor, ...] | list[Tensor]) -> Tensor:
    """ Flatten a sequence of batched tensors.

    Args:
        pointclouds (Sequence): Sequence of B batches [(M, N, *), ... (M, N, *)].

    Returns:
        Batch of point clouds of length (B * M, *).
    """
    return flatten(torch.stack(pointclouds))  # (B, M, N, 3) -> (B * M, N, 3)


def expand_and_flatten(pointclouds: Tensor, k: int) -> Tensor:
    """ Repeat k times each element in a batch or sequence of point clouds.

    Args:
        pointclouds (Sequence): Batch of point clouds `(B, N, *)`.
        k (int): Number of repetitions.

    Returns:
        Batch of repeated point clouds `(k * B, N, *)`.
    """
    return flatten(pointclouds.unsqueeze(1).expand(-1, k, -1, -1))
