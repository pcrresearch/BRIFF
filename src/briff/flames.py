from __future__ import annotations
from typing import Callable, cast
from tqdm.auto import tqdm
import torch
from torch import Tensor
from pytorch3d import transforms as T
from pytorch3d.loss import chamfer_distance
from .so3 import SO3
from .pcd_utils import flatten, unflatten, expand_and_flatten
import nvidia_smi  # nvidia-ml-py3


# _________________________________________________________________________________________________________ #

def bytes2human(number: int, decimal_unit: bool = True) -> str:
    """ Convert number of bytes in a human readable string.
    >>> bytes2human(10000, True)
    '10.00 KB'
    >>> bytes2human(10000, False)
    '9.77 KiB'
    Args:
        number (int): Number of bytes
        decimal_unit (bool): If specified, use 1 kB (kilobyte)=10^3 bytes.
            Otherwise, use 1 KiB (kibibyte)=1024 bytes
    Returns:
        str: Bytes converted in readable string
    """
    symbols = ['K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    symbol_values = [(symbol, 1000 ** (i + 1) if decimal_unit else (1 << (i + 1) * 10))
                     for i, symbol in enumerate(symbols)]
    for symbol, value in reversed(symbol_values):
        if number >= value:
            suffix = "B" if decimal_unit else "iB"
            return f"{float(number)/value:.2f}{symbol}{suffix}"
    return f"{number} B"


# _________________________________________________________________________________________________________ #

def pointwise_single_directional_chamfer_distance(X: Tensor, Y: Tensor) -> Tensor:
    d, _ = chamfer_distance(X, Y, batch_reduction=None, point_reduction=None, single_directional=True)
    return cast(Tensor, d)


def chamfer_distances(X: Tensor, Y: Tensor, bidirectional: bool = False, trim_ratio: float = 0.) -> Tensor:
    d_x2y = pointwise_single_directional_chamfer_distance(X, Y)  # (B, N)
    if trim_ratio > 0:
        K = int((1 - trim_ratio) * X.shape[1])  # K = (1 - trim_ratio) * N_S
        d_x2y = d_x2y.sort().values[:, :K]  # (B, K)
    d_x2y = d_x2y.mean(dim=1)  # (B,)
    if not bidirectional:
        return d_x2y
    d_y2x = pointwise_single_directional_chamfer_distance(Y, X)
    if trim_ratio > 0:
        K = int((1 - trim_ratio) * Y.shape[1])  # K = (1 - trim_ratio) * N_M
        d_y2x = d_y2x.sort().values[:, :K]  # (B, K)
    d_y2x = d_y2x.mean(dim=1)  # (B,)
    return (d_x2y + d_y2x) / 2


def chamfer_distances_between_views_and_model(
    *, views: Tensor, model: Tensor, from_model: bool, bidirectional: bool, trim_ratio
) -> Tensor:
    model_expanded = model.expand(len(views), -1, -1)
    inputs = (model_expanded, views) if from_model else (views, model_expanded)
    return chamfer_distances(*inputs, bidirectional, trim_ratio)


# _________________________________________________________________________________________________________ #

def chunked_so3_orbits_and_criteria_B_to_B(
    X: Tensor,
    Y: Tensor,
    criterion: Callable,
    so3: SO3,
    chunk_size: int,
    batch_size: int,
    progress_bar: bool,
    monitor_vram: bool,
    gpu_index: int,
) -> Tensor:
    """ B unique sources (B, N_s, 3), B unique models (B, N_m, 3). Used for batched pairwise registration.
        X & Y do not represent sources and models as this function can be used to compute oriented losses
        from sources to models or from models to sources.
    """
    if monitor_vram:
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)
    losses = list()
    if progress_bar:
        pbar = tqdm(X.split(chunk_size))
        iterable = zip(pbar, Y.split(chunk_size))
    else:
        iterable = zip(X.split(chunk_size), Y.split(chunk_size))
    for X_chunk, Y_chunk in iterable:
        # X_chunk & Y_chunk: (C, N, 3) (C=chunk_size)
        # L denotes the number of rotations sampled (i.e. len(so3))
        if progress_bar and monitor_vram:
            gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            pbar.set_postfix_str(f'VRAM: {bytes2human(gpu_info.used)}')
        X_orbits = torch.stack([so3.transforms.transform_points(x) for x in X_chunk])  # (C, L, N, 3)
        X_orbits_flatten = X_orbits.view(len(X_chunk) * len(so3), -1, 3)  # (C * L, N, 3)
        Y_chunk_expanded_flatten = expand_and_flatten(Y_chunk, len(so3))  # (C, N, 3) -> (C * L, N, 3)
        s_batches = X_orbits_flatten.split(batch_size)
        m_batches = Y_chunk_expanded_flatten.split(batch_size)
        losses_chunk = torch.hstack([criterion(x, y) for x, y in zip(s_batches, m_batches)])
        losses_chunk = unflatten(losses_chunk, (len(X_chunk), len(so3)))
        losses.append(losses_chunk)
    losses = torch.vstack(losses)
    return losses


def chunked_so3_orbits_and_criteria_B_to_1(
    views: Tensor,
    model: Tensor,
    criterion: Callable,
    so3: SO3,
    chunk_size: int,
    batch_size: int,
    progress_bar: bool,
    monitor_vram: bool,
    gpu_index: int,
) -> Tensor:
    """ model is a single point cloud (N, 3) and views is a batch (B, N, 3). Used for generative
        multiviews registration.
        The provided criterion MUST accept two keyword arguments 'model' and 'view'.
    """
    if monitor_vram:
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_index)
    losses = list()
    pbar = tqdm(views.split(chunk_size)) if progress_bar else views.split(chunk_size)
    for views_chunk in pbar:
        if progress_bar and monitor_vram:
            gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            pbar.set_postfix_str(f'VRAM: {bytes2human(gpu_info.used)}')
        views_chunk_orbits = torch.stack([so3.transforms.transform_points(v) for v in views_chunk])
        views_chunk_orbits_flatten = views_chunk_orbits.view(len(views_chunk) * len(so3), -1, 3)
        views_batches = views_chunk_orbits_flatten.split(batch_size)
        losses_chunk = torch.hstack([criterion(views=v, model=model) for v in views_batches])
        losses_chunk = unflatten(losses_chunk, (len(views_chunk), len(so3)))
        losses.append(losses_chunk)
    losses = torch.vstack(losses)
    return losses


def chunked_so3_orbits_and_criteria(
    X: Tensor,
    Y: Tensor,
    criterion: Callable,
    so3: SO3,
    chunk_size: int,
    batch_size: int,
    progress_bar: bool,
    monitor_vram: bool,
    gpu_index: int,
) -> Tensor:
    """ There are 2 setups:
        - X & Y are two batches of point clouds (B, N, 3):
            typically for batched pairwise registration
        - X is a batch (B, N, 3) (views) and Y is a single point cloud (N, 3) (model):
            typically for generative multiviews registration.
    """
    params = (criterion, so3, chunk_size, batch_size, progress_bar, monitor_vram, gpu_index)
    if X.ndim != 3:
        raise ValueError('The first argument must be a batch of point clouds (B, N, 3).')
    if Y.ndim == 3:  # criteria between two batches
        return chunked_so3_orbits_and_criteria_B_to_B(X, Y, *params)
    # criteria between a batch of views and a single model
    return chunked_so3_orbits_and_criteria_B_to_1(X, Y, *params)

# _________________________________________________________________________________________________________ #


def flames_from_losses(
    losses: Tensor,
    sources: Tensor,
    so3: SO3,
    J: int,
    parallel_search: bool,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    # 1. Batched local minima search
    lms = list()
    for l in losses.split(batch_size):
        lms.extend(so3.local_minima_search(l, J, multiviews=True, parallel=parallel_search))
    # 2. Repeat if required and stack minima
    lms_stack = list()
    for l in lms:
        k = len(l)
        if k < J:  # repeat first minimum if not enough minima, so that we can stack
            n = J - k
            best_minimum = torch.as_tensor(l[0])
            l = torch.hstack((best_minimum.expand(n), torch.as_tensor(l)))
        lms_stack.append(l)
    lms_stack = torch.stack(lms_stack)
    # 3. Retrieve rotations from local minima and apply them
    rotation_starts = so3.sampling[lms_stack]
    rotation_starts_flatten = flatten(rotation_starts)
    sources_expanded_flatten = expand_and_flatten(sources, J)
    sources_starts = T.Rotate(R=rotation_starts_flatten).transform_points(sources_expanded_flatten)
    sources_starts = unflatten(sources_starts, (len(sources), J))
    return sources_starts, rotation_starts


def flames(
    X: Tensor,
    Y: Tensor,
    *,
    so3: SO3,
    J: int,
    criterion: Callable,
    parallel_search: bool,
    chunk_size: int,
    batch_size: int,
    progress_bar: bool,
    monitor_vram: bool,
    gpu_index: int,
) -> tuple[Tensor, Tensor]:
    """ There are 2 setups:
        - X & Y are two batches of point clouds (B, N, 3):
            typically for batched pairwise registration
        - X is a batch (B, N, 3) (views) and Y is a single point cloud (N, 3) (model):
            typically for generative multiviews registration.
    """
    so3 = so3.to(X.device)
    # 1. SO(3) orbits and criteria
    params = (so3, chunk_size, batch_size, progress_bar, monitor_vram, gpu_index)
    losses = chunked_so3_orbits_and_criteria(X, Y, criterion, *params)
    # 2. Local Minima Search
    lms_batch_size = 16 * chunk_size  # should be a safe bet
    return flames_from_losses(losses, X, so3, J, parallel_search, lms_batch_size)
