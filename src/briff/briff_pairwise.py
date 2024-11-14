""" In this version, I expect all point clouds to be of same length. This allows me to batch all operations,
    in particular for the SO(3) local minima search. I believe overall it provides ~ 30% speed improvement.
    Batching everything while allowing varying inputs shapes is feasible, typically through the message
    passing scheme employed in torch geometric (aka graph processing). I didn't implement that though.
"""

from __future__ import annotations
from typing import Callable, Optional, Any, cast
from typing_extensions import TypeAlias
from functools import partial
from time import perf_counter
import torch
from torch import Tensor
from pytorch3d import transforms as T
from pytorch3d.ops.utils import eyes
from .so3 import SO3
from .pcd_utils import center, pairwise_max_norm, flatten, unflatten, expand_and_flatten
from .flames import chamfer_distances, flames


# _________________________________________________________________________________________________________ #

def timed(method: Callable) -> Callable:
    def wrapper(self, *args, **kwargs) -> Any:
        tic = perf_counter()
        out = method(self, *args, **kwargs)
        toc = perf_counter()
        self.elapsed_time[method.__name__] = toc - tic
        return out
    return wrapper


# _________________________________________________________________________________________________________ #

Losses: TypeAlias = Tensor
RigidMotion: TypeAlias = Tensor
PointClouds: TypeAlias = Tensor

RegistrationFn: TypeAlias = Callable[[PointClouds, PointClouds], RigidMotion]
Criterion: TypeAlias = Callable[[PointClouds, PointClouds], Losses]


# _________________________________________________________________________________________________________ #

class PairwiseBRIFF:

    def __init__(
        self,
        # Core
        registration_fn: RegistrationFn,
        criterion: Optional[Criterion] = None,
        # Chamfer
        bidirectional: bool = False,
        trim_ratio: float = 0.,
        # SO(3) Local Minima Search
        L: int = 20_000,
        K: int = 64,
        J: int = 64,
        # VRAM usage
        chunk_size: int = 8,
        batch_size: int = 100_000,
        parallel_search: bool = True,
        # Logging
        verbose: bool = True,
        progress_bar: bool = False,
        monitor_vram: bool = False,
        gpu_index: int = 0,
    ) -> None:
        """ Setup the BRIFF algorithm for pairwise registration.

        Args:
            registration_fn (Callable):
                Base registration function upon which BRIFF will be executed. It MUST have the following
                signature:
                    `T_hat: Tensor = registration_fn(sources: Tensor, models: Tensor)`
                where:
                    - `sources` and `models` are batches of point clouds `(B, N, 3)`.
                    - `T_hat_batched` is a batch of rigid motions `(B, 4, 4)`.
            criterion (Callable, optional):
                Scalar valued function to evaluate the quality of a registration. If provided, it MUST have
                the following signature:
                    `losses = criterion(registered_sources, models)`
                where:
                    - `registered_sources` and `models` are tensors of of point clouds batches `(B, N, 3)`.
                    - `T_hat_batched` is a batch of rigid motions `(B, 4, 4)`.
                If `None`, BRIFF will use the Chamfer distance. Defaults to `None`.
            bidirectional (str):
                Only used when criterion is None. Direction of the Chamfer distances. If `True`, compute both
                sources to models and models to sources, otherwise compute sources to models only.
                Defaults to `False`.
            trim_ratio (float, optional):
                Trimming ratio of the initial Chamfer distances. Defaults to `0`.
            L (int, optional):
                Number of rotations uniformly sampled in SO(3). Defaults to `20_000`.
            K (int, optional):
                Number of neighbors to consider for the SO(3) local minima search. Defaults to `64`.
            J (int, optional):
                Maximal number of local minima to consider per source. Defaults to `64`.
            chunk_size (int, optional):
                Number of views being handled in parallel. Controls memory consumption. VRAM usage can be
                monitored with progress_bar=True and monitor_vram=True. Defaults to 4.
            batch_size (int, optional): Number of Chamfer distances being computed in parallel. Controls
                memory consumption. VRAM usage can be monitored with progress_bar=True and monitor_vram=True.
                Defaults to 100_000.
            parallel_search (bool, optional):
                If `True`, search local minima for all views in parallel. Linearly faster (w.r.t the number
                of point clouds), but consumes a lot more memory. Defaults to `True`.
            verbose (bool, optional):
            If `True`, display BRIFF step advancements. Defaults to `True`.
            progress_bar (bool):
                If True, display progress bars for the SO(3) local minima search. Defaults to `False`.
        """
        self.registration_fn = registration_fn
        self.J = J
        self.verbose = verbose
        self.progress_bar = progress_bar
        self.so3 = SO3(L, K, verbose=verbose)
        self.criterion = criterion if criterion is not None else partial(chamfer_distances,
                                                                         bidirectional=bidirectional,
                                                                         trim_ratio=trim_ratio)
        self.flames = partial(flames,
                              so3=self.so3,
                              J=J,
                              criterion=self.criterion,
                              parallel_search=parallel_search,
                              chunk_size=chunk_size,
                              batch_size=batch_size,
                              progress_bar=progress_bar,
                              monitor_vram=monitor_vram,
                              gpu_index=gpu_index)
        self.elapsed_time = {}
        PairwiseBRIFF.check_params(L, K, trim_ratio)

    def __str__(self) -> str:
        algo = self.registration_fn
        crit = self.criterion
        algo_name = algo.func.__name__ if isinstance(algo, partial) else algo.__name__
        crit_name = crit.func.__name__ if isinstance(crit, partial) else crit.__name__
        string = (f'{self.__class__.__name__}:\n'
                  f'L={self.so3.L:,} | K={self.so3.K} | J={self.J}\n'
                  f'algo={algo_name}\n'
                  f'criterion={crit_name}')
        return string

    def print_if_verbose(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def check_params(L: int, K: int, trim_ratio: float) -> None:
        # TODO: update checks: can we only check resolution / radius ratio ?
        # 1. Check trimming and resolution
        if trim_ratio > 0 and L >= 1e4:
            print((f'[WARNING] You selected {trim_ratio=} with {L=}. With trimmming, lower values of L '
                   'usually work better and faster. Try to reduce L & K.'))
        # 2. Check radius
        radius = L / K
        radius_too_small, radius_too_large = radius < 150, radius > 250
        if radius_too_small or radius_too_large:
            radius_issue = 'large' if radius_too_large else 'small'
            correct = 'decrease' if radius_too_large else 'increase'
            print(f'[WARNING] You selected {L=} & {K=}. The ratio L/K seems too {radius_issue}. If you get '
                  f'bad results, try to {correct} this ratio.')

    # _____________________________________________________________________________________________________ #

    def preprocess(self, S: Tensor, M: Tensor) -> None:
        """ Center & Pairwise Max Norm """
        self.N = len(S)
        self.original_sources = S
        self.sources, self.models = pairwise_max_norm(center(S), center(M))

    @timed
    def initialize(self) -> None:
        """ Top K SO(3) local minima search. """
        sources_starts, self.rotations_starts = self.flames(self.sources, self.models)
        self.sources_starts = flatten(sources_starts)                 # (B, M, N, 3) -> (B * M, N, 3)
        self.models_starts = expand_and_flatten(self.models, self.J)  # (B, N, 3) -> (B * M, N, 3)

    @timed
    def run_registration(self) -> None:
        """ Run local refinement registration algorithm."""
        self.print_if_verbose(f':: Running {self.N * self.J:,} pairwise registrations ...')
        self.M_sf2m_hat = self.registration_fn(self.sources_starts, self.models_starts)

    @timed
    def get_best_minimum(self) -> None:
        """ Get best out of K minima. """
        self.print_if_verbose(':: Retrieving best pairwise results.')
        # 1. Apply estimated motions & compute all losses
        sf_hat = T.Transform3d(matrix=self.M_sf2m_hat).transform_points(self.sources_starts)  # (B * M, N, 3)
        losses = self.criterion(sf_hat, self.models_starts)                     # (B * M)
        # 2. Unflatten
        losses = unflatten(losses, (self.N, self.J))                         # (B, M)
        M_sf2m_hat = unflatten(self.M_sf2m_hat, (self.N, self.J))            # (B, M)
        sf_hat = unflatten(sf_hat, (self.N, self.J))                         # (B, M, N, 3)
        # 3. Retrieve best minimum per pair
        best_minima_indices = losses.argmin(dim=1)                              # (B,)
        # 4. From best minimum, get best estimated motion, best rotation start, best registered sources
        M_sf2m_hat = M_sf2m_hat[torch.arange(self.N), best_minima_indices]      # (B, 4, 4)
        R_f = self.rotations_starts[torch.arange(self.N), best_minima_indices]  # (B, 3, 3)
        # ! self.sources is aligned with self.models, not self.original_models
        self.sources = sf_hat[torch.arange(self.N), best_minima_indices]        # (B, N, 3)
        self.T_sf2m_hat = T.Transform3d(matrix=M_sf2m_hat)
        self.T_f = T.Rotate(R=R_f)

    def compose_motions(self) -> None:
        """ Compose SO(3) local minima rotation and local refinement motion. """
        self.T_s2m_hat = self.T_f.compose(self.T_sf2m_hat)
        self.M_s2m_hat = self.T_s2m_hat.get_matrix()

    def run_pairwise(self) -> None:
        self.initialize()
        self.run_registration()
        self.get_best_minimum()
        self.compose_motions()

    def run_pairwise_iterated(self, S: Tensor, M: Tensor, iter: int) -> Tensor:
        """ Pairwise BRIFF algorithm: registrate sources onto models.

        Args:
            S (Tensor): Batch of source point clouds of same sizes: (B, N_s, 3).
            M (Tensor): Batch of model point clouds of same sizes: (B, N_m, 3).
        """
        self.preprocess(S, M)  # center & normalize once and for all
        T_s2m_hat_iterated = T.Transform3d(matrix=eyes(4, len(S), device=S.device))
        for i in range(iter):
            if iter > 1:
                self.print_if_verbose(f':: Iter {i+1}/{iter}')
            self.run_pairwise()
            T_s2m_hat_iterated = T_s2m_hat_iterated.compose(self.T_s2m_hat)
        self.T_s2m_hat = T_s2m_hat_iterated
        self.M_s2m_hat = self.T_s2m_hat.get_matrix()
        return self.M_s2m_hat

    def register(self, S: Optional[Tensor], M: Optional[Tensor], iter: int = 1) -> tuple[Tensor, Tensor]:
        motions_estimated = hasattr(self, 'M_s2m_hat')
        if motions_estimated:
            return self.sources, self.models
        if not motions_estimated and (S is None or M is None):
            missing_data_string = 'sources' if S is None else 'models'
            raise ValueError((f"You didn't provide {missing_data_string} but no motions have been "
                              "estimated yet. You most likely forgot to run estimation first."))
        S, M = cast(Tensor, S), cast(Tensor, M)
        self.run_pairwise_iterated(S, M, iter)
        return self.sources, self.models

    def __call__(self, S: Tensor, M: Tensor, iter: int = 1) -> Tensor:
        """ This function returns the estimated transform. Call self.register() to get a tuple of aligned
            sources and models.
            Once finished, runtimes can be accessed through the self.elapsed_time dictionary.
        """
        return self.run_pairwise_iterated(S, M, iter)
