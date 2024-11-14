from __future__ import annotations
from typing import Optional
from typing_extensions import TypeAlias
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import pytorch3d.transforms as T


SO3KnnGraph: TypeAlias = "dict[int, dict[str, Tensor]]"


# +-------------------------------------------------------------------------------------------------------+ #
# |                                     FAST SO(3) GEODESIC DISTANCES                                     | #
# +-------------------------------------------------------------------------------------------------------+ #


def triangular_to_square_matrix_indices(i: int, N: int) -> tuple[int, int]:
    """ If the upper triangular part, without the diagonal, of a square matrix (N, N) has been flatten
        into a tensor of length N * (N - 1) / 2, this function takes the size N of the original matrix, an
        index in the flatten tensor, and return the corresponding row and column indices in the original
        square matrix.
    """
    row = 0
    while i >= (N - 1 - row):
        i -= (N - 1 - row)
        row += 1
    col = row + 1 + i
    return row, col


def all_triangular_to_square_matrix_indices(N: int) -> tuple[Tensor, Tensor]:
    """ If the upper triangular part, without the diagonal, of a square matrix M (N, N) has been flatten
        into a 1D tensor T of length N * (N-1) / 2, return two 1D tensors I and J of length K = N (N - 1) / 2
        such that for k=1...K, I[k] and J[k] are the indices satisfying T[k] = M[I[k]][J[k]].
    """
    K = N * (N - 1) / 2
    I, J = list(), list()
    for k in range(int(K)):
        i, j = triangular_to_square_matrix_indices(k, N)
        I.append(torch.as_tensor(i))
        J.append(torch.as_tensor(j))
    I, J = torch.stack(I), torch.stack(J)
    return I, J


def pairwise_so3_relative_angles(R_gt: Tensor, R_hat: Tensor) -> Tensor:
    """ If N rotations have been estimated to align N views in an arbitrary orientation, and we need to
        compute the estimation error, we have to compute the pairwise SO(3) geodesic distance between
        R_hat[i]R_gt[i] and R_hat[j]R_gt[j], resulting in N(N-1)/2 distances. This function computes
        these N(N-1)/2 distances in parallel.
    Args:
        R_gt (Tensor): Groundtruth poses (N, 3, 3).
        R_hat (Tensor): Estimated poses (N, 3, 3).

    Returns:
        Tensor: Pairwise rotation estimation error (N(N-1)/2,).
    """
    assert len(R_gt) == len(R_hat)
    N = len(R_gt)
    I, J = all_triangular_to_square_matrix_indices(N)
    R1 = R_hat[I] @ R_gt[I]
    R2 = R_hat[J] @ R_gt[J]
    return T.so3_relative_angle(R1, R2) * 180 / torch.pi


# +-------------------------------------------------------------------------------------------------------+ #
# |                                            SO(3) SAMPLING                                             | #
# +-------------------------------------------------------------------------------------------------------+ #

def super_fibonacci_spirals(n: int | float, double: bool = False, as_matrix: bool = False) -> Tensor:
    """ Generate n samples on SO(3) as unit quaternions, then convert them to Euler's angles.
    Alexa M., Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO3, CVPR 2022.
    """
    n = int(n)
    phi = np.sqrt(2.0)
    psi = 1.533751168755204288118041
    Q = np.empty(shape=(n, 4), dtype=np.float64 if double else np.float32)
    i = np.arange(n)
    s = i + 0.5
    r = np.sqrt(s / n)
    R = np.sqrt(1.0 - s / n)
    alpha = 2.0 * np.pi * s / phi
    beta = 2.0 * np.pi * s / psi
    Q[i, 0] = r * np.sin(alpha)
    Q[i, 1] = r * np.cos(alpha)
    Q[i, 2] = R * np.sin(beta)
    Q[i, 3] = R * np.cos(beta)
    matrices = T.quaternion_to_matrix(torch.tensor(Q))
    if as_matrix:
        return matrices
    angles = T.matrix_to_euler_angles(matrices, 'ZXZ')
    return angles


# +-------------------------------------------------------------------------------------------------------+ #
# |                                 SO(3) KNN GRAPH & LOCAL MINIMA SEARCH                                 | #
# +-------------------------------------------------------------------------------------------------------+ #

def make_so3_knn_graph(so3: Tensor, K: int = 8) -> tuple[Tensor, Tensor]:
    """ Note that each rotation is NOT its own neighbor. """
    if not so3.is_cuda and torch.cuda.is_available():
        print((
            f"[WARNING] Requested SO(3) k-nn graph computation (L={len(so3)}, K={K}) on CPU but a CUDA GPU"
            " is available. I'll attempt to use it."
        ))
        so3 = so3.cuda()
    all_indices, all_relative_angles = list(), list()
    R2 = so3
    for R in so3:
        R1 = R.expand_as(R2)
        relative_angles = T.so3_relative_angle(R1, R2) * 180 / torch.pi
        values, indices = relative_angles.sort()
        all_indices.append(indices[1: K + 1].cpu())
        all_relative_angles.append(values[1: K + 1].cpu())
    return torch.stack(all_indices), torch.stack(all_relative_angles)


def save_so3_knn_graph(
    indices: Tensor, relative_angles: Tensor, L: int, K: int, output_dir: str = ".cache/"
) -> None:
    """ Store two tensors Indices and Angles such that:
        for all i in {1, ..., L}
        R[i]'s neighbors are R[Indices[i]], and relative angles between R[i] and R[Indices[i]] are
        Angles[i].
    """
    here = Path(__file__).resolve().parent
    output_path = here / output_dir
    output_path.mkdir(exist_ok=True)
    filename = f"so3_knn_lookup_table_indices_L={L}_K={K}.pt"
    torch.save(indices, output_path / filename)
    filename = f"so3_knn_lookup_table_angles_L={L}_K={K}.pt"
    torch.save(relative_angles, output_path / filename)


def make_and_save_so3_knn_lookup_table(L: int, K: int, output_dir: str = ".cache/") -> None:
    here = Path(__file__).resolve().parent
    output_path = here / output_dir
    output_path.mkdir(exist_ok=True)
    filename = f"so3_knn_lookup_table_L={L}_K={K}.pt"
    output_path = output_path / filename
    so3 = super_fibonacci_spirals(L, as_matrix=True)
    indices, relative_angles = make_so3_knn_graph(so3, K)
    save_so3_knn_graph(indices, relative_angles, L, K, output_dir)


def get_so3_knn_graph(
    L: int | float,
    K: int,
    verbose: bool = False,
    output_dir: str = ".cache/"
) -> SO3KnnGraph:
    """ Search cached files on disk. If not found, create them. """
    L = int(L)
    here = Path(__file__).resolve().parent
    output_path = here / output_dir
    filename = f"so3_knn_lookup_table_indices_L={L}_K={K}.pt"
    if not (output_path / filename).exists():
        if verbose:
            print((":: Cached SO(3) Lookup Table not found. It will be created on the fly and saved for "
                   "later use. This will occur extra runtime."))
        make_and_save_so3_knn_lookup_table(L, K, output_dir)
    elif verbose:
        print(":: Using cached SO(3) knn-graph.")
    path = output_path / f"so3_knn_lookup_table_indices_L={L}_K={K}.pt"
    indices = torch.load(path)
    path = output_path / f"so3_knn_lookup_table_angles_L={L}_K={K}.pt"
    angles = torch.load(path)
    knn_graph = {i: {'indices': indices[i], 'relative_angles': angles[i]} for i in range(len(indices))}
    return knn_graph


def single_view_find_so3_local_minima_from_knn_graph(so3_knn_graph: SO3KnnGraph, values: Tensor) -> Tensor:
    neighbors = torch.stack([n['indices'] for n in so3_knn_graph.values()])  # type: ignore
    local_minima_indices = (values[:, None] <= values[neighbors]).all(dim=1).argwhere().squeeze(dim=1)
    return local_minima_indices[values[local_minima_indices].argsort()]


def parallel_find_so3_local_minima_from_knn_graph(so3_knn_graph: SO3KnnGraph, values: Tensor) -> list[Tensor]:
    """ All views in parallel: values is a tensor (num_views, so3_sampling_size). """
    neighbors = torch.stack([n['indices'] for n in so3_knn_graph.values()])  # type: ignore
    # boolean (N, L) =   (N, L, 1)     <=       (L, K)         (all)
    local_minima = (values[:, :, None] <= values[:, neighbors]).all(dim=2)
    local_minima_indices = [l.argwhere().squeeze(dim=1) for l in local_minima]
    local_minima_indices_sorted = [l[v[l].argsort()] for l, v in zip(local_minima_indices, values)]
    return local_minima_indices_sorted


def find_so3_local_minima_from_knn_graph(
    so3_knn_graph: SO3KnnGraph,
    values: Tensor,
    parallel: bool
) -> list[Tensor]:
    if parallel:
        return parallel_find_so3_local_minima_from_knn_graph(so3_knn_graph, values)
    return [single_view_find_so3_local_minima_from_knn_graph(so3_knn_graph, v) for v in values]


# +-------------------------------------------------------------------------------------------------------+ #
# |                                       SO(3) CONTAINER STRUCTURE                                       | #
# +-------------------------------------------------------------------------------------------------------+ #

class SO3:

    def __init__(
        self,
        L: int,
        K: int,
        device: Optional[torch.device] = None,
        verbose: bool = True
    ) -> None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L = int(L)
        self.K = int(K)
        self.sampling = super_fibonacci_spirals(L, as_matrix=True).to(device).detach()
        self.transforms = T.Rotate(R=self.sampling)
        self.knn_graph = get_so3_knn_graph(L, K, verbose)
        self.local_minima_history: list[list[Tensor]] = list()

    def __len__(self) -> int:
        return self.L

    def nearest_neighbor_distances(self) -> Tensor:
        return torch.hstack([v['relative_angles'][0] for v in self.knn_graph.values()])

    def farthest_neighbor_distances(self) -> Tensor:
        return torch.hstack([v['relative_angles'][-1] for v in self.knn_graph.values()])

    def local_minima_search(
        self, values: Tensor, J: int, multiviews: bool, parallel: bool
    ) -> Tensor | list[Tensor]:
        if not multiviews:
            return single_view_find_so3_local_minima_from_knn_graph(self.knn_graph, values)[:J]
        local_minima = find_so3_local_minima_from_knn_graph(self.knn_graph, values, parallel)
        self.local_minima_history.append(local_minima)
        return [lm[:J] for lm in local_minima]

    def to(self, device: str | torch.device) -> SO3:
        self.sampling = self.sampling.to(device)
        self.transforms = self.transforms.to(device)
        self.knn_graph = {i: {'indices': neighborhood['indices'].to(device),
                              'relative_angles': neighborhood['relative_angles'].to(device)}
                          for i, neighborhood in self.knn_graph.items()}
        return self
