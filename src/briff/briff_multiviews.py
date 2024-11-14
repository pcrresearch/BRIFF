from __future__ import annotations
from typing import Callable, Any
from abc import ABC, abstractmethod
from time import perf_counter
from functools import partial
from tqdm.auto import tqdm
import torch
from torch import Tensor
from pytorch3d.transforms import Rotate, Transform3d, so3_relative_angle
from .so3 import SO3, pairwise_so3_relative_angles
from .pcd_utils import center, max_norm
from .flames import chamfer_distances_between_views_and_model, flames
from .briff_pairwise import PairwiseBRIFF


# _________________________________________________________________________________________________________ #

def timed(method: Callable) -> Callable:
    def wrapper(self, *args, **kwargs) -> Any:
        tic = perf_counter()
        out = method(self, *args, **kwargs)
        toc = perf_counter()
        if method.__name__ in self.elapsed_time:
            self.elapsed_time[method.__name__].append(toc - tic)
        else:
            self.elapsed_time[method.__name__] = [toc - tic]
        return out
    return wrapper


# _________________________________________________________________________________________________________ #

class GenerativeMultiviewsMethod(ABC):

    """ Abstract class that generative multiviews registration methods must implement in order to be used
        with BRIFF.
    """

    model: Tensor

    @abstractmethod
    def criterion(self, *, views: Tensor, model: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def optimize(self, views: Tensor, generative: bool = False) -> Tensor:
        raise NotImplementedError


# _________________________________________________________________________________________________________ #

class GenerativeMultiviewsBRIFF(PairwiseBRIFF):

    def __init__(
        self,
        base_method: GenerativeMultiviewsMethod,
        from_model: bool = False,
        bidirectional: bool = True,
        R_eps: int | float = 5,
        max_iter: int = 10,
        trim_ratio: float = 0.,
        L: int = 50_000,
        K: int = 256,
        J: int = 16,
        chunk_size: int = 4,
        batch_size: int = 100_000,
        parallel_search: bool = True,
        verbose: bool = True,
        progress_bar: bool = False,
        monitor_vram: bool = False,
        gpu_index: int = 0,
    ) -> None:
        """ Setup the BRIFF algorithm for generative multiviews registration.

        Args:
            base_method (GenerativeMultiviewsMethod):
                Generative multiviews registration method upon which BRIFF will be based. Must inherit from
                `GenerativeMultiviewsMethod`.
            from_model (bool, optional):
                Chamfer distances direction for initialization. Defaults to `False`.
            bidirectional (bool, optional):
                Chamfer distances direction for initialization. If `True`, compute both sources to model and
                model to sources, otherwise compute sources to models only. Defaults to `False`.
                Defaults to `True`.
            R_eps (int | float, optional):
                Relative angle threshold (in degree) between old and new rotations to detect escapes from
                local minima. Defaults to `30`.
            max_iter (int, optional):
                Maximal number of iterations of joint generative optimization and non generative pairwise
                restarts. Defaults to `10`.
            trim_ratio (float, optional):
                Trimming ratio of the initial Chamfer distances. Defaults to `0`.
            L (int, optional):
                Number of rotations uniformly sampled in SO(3). Defaults to `50_000`.
            K (int, optional):
                Number of neighbors to consider for the SO(3) local minima search. Defaults to `256`.
            J (int, optional):
                Maximal number of local minima to consider per source. Defaults to `16`.
            chunk_size (int, optional):
                Number of views being handled in parallel. Controls memory consumption. VRAM usage can be
                monitored with `progress_bar=True` and `monitor_vram=True`. Defaults to `4`.
            batch_size (int, optional): Number of Chamfer distances being computed in parallel. Controls
                memory consumption. VRAM usage can be monitored with `progress_bar=True` and
                `monitor_vram=True`. Defaults to `100_000`.
            parallel_search (bool, optional):
                If `True`, search local minima for all views in parallel. Linearly faster (w.r.t the number
                of point clouds), but consumes a lot more memory. Defaults to `True`.
            verbose (bool, optional):
                If `True`, display BRIFF step advancements. Defaults to `True`.
            progress_bar (bool):
                If True, display progress bars for the SO(3) local minima search. Defaults to `False`.
            monitor_vram (bool, optional):
                If progress_bar is True, will display the current VRAM usage of the GPU (specified by
                `gpu_index`). Defaults to `False`.
            gpu_index (int, optional):
                Index of the GPU to monitor. Defaults to `0`.
        """
        self.base_method = base_method
        self.from_model = from_model
        self.J = J
        self.R_eps = R_eps
        self.max_iter = max_iter
        self.verbose = verbose
        self.progress_bar = progress_bar
        so3 = SO3(L, K, verbose=verbose)
        self.flames = partial(flames,
                              so3=so3,
                              J=J,
                              parallel_search=parallel_search,
                              chunk_size=chunk_size,
                              batch_size=batch_size,
                              progress_bar=progress_bar,
                              monitor_vram=monitor_vram,
                              gpu_index=gpu_index)
        self.chamfer_distances = partial(chamfer_distances_between_views_and_model,
                                         from_model=from_model,
                                         bidirectional=bidirectional,
                                         trim_ratio=trim_ratio)
        self.elapsed_time = {}

    @property
    def R_hat(self) -> Tensor:
        return self.T_hat.get_matrix()[:, :3, :3]

    @timed
    def initialize(self, views: Tensor) -> None:
        """ Chamfer distances between views[0] and views[1:], followed by Local Minima Search, and best
            minimum (per view) retrieval.
        """
        self.print_if_verbose(':: Running top 1 SO(3) local minima search with Chamfer criterion')
        self.num_views = len(views)
        # 1. Initial setup
        self.original_views = max_norm(center(views))
        model, sources = self.original_views[0], self.original_views[1:]
        # 2. FLAMES best minimum
        sources_starts, R_starts = self.flames(sources, model, criterion=self.chamfer_distances)
        sources_starts = [s[0] for s in sources_starts]
        R_flames_init = torch.stack((torch.eye(3, device=model.device), *[R[0] for R in R_starts]))
        V_flames_init = (model, *sources_starts)
        V_flames_init = torch.stack(V_flames_init).clone()
        T_hat = Rotate(R=R_flames_init).inverse().clone()
        self.base_method.model = model
        self.views, self.T_hat = V_flames_init, T_hat

    def update_estimation(self, new_T: Transform3d) -> None:
        """ Compose newly estimated motions with current estimations. """
        self.T_hat = new_T.inverse().compose(self.T_hat)
        self.views = self.T_hat.inverse().transform_points(self.original_views)

    @timed
    def joint_generative_optimization(self) -> None:
        """ Apply the base generative multiviews method and center the obtained model. """
        self.print_if_verbose(':: Running joint generative optimization')
        T_hat = self.base_method.optimize(self.views, generative=True)
        center = self.base_method.model.mean(dim=0)[None]  # (1, 3)
        self.base_method.model -= center
        self.update_estimation(Transform3d(matrix=T_hat).translate(-center))

    @timed
    def pairwise_non_generative_restarts(self) -> None:
        """ Apply the base method in a non generative setup: for each of the N views, compute the J best
            local minima between the view and the current model estimation, and run N * K non generative
            pairwise registrations.
        """
        self.print_if_verbose(':: Running pairwise non-generative restarts')
        views_starts, rotations_starts = self.flames(self.views, self.base_method.model,
                                                     criterion=self.base_method.criterion)
        T_hat_pairwise_restart = list()
        iterable = zip(tqdm(rotations_starts) if self.progress_bar else rotations_starts, views_starts)
        for R, view_starts in iterable:
            T_hats = self.base_method.optimize(view_starts, generative=False)
            T_hats = Transform3d(matrix=T_hats)  # (k, 4, 4)
            views_hat = T_hats.transform_points(view_starts)  # (k, n_s, 3)
            losses = self.base_method.criterion(views=views_hat)
            idx_best_local_minimum = losses.argmin()
            T_hat_pairwise_restart_i = Rotate(R=R).compose(T_hats).get_matrix()[idx_best_local_minimum]
            T_hat_pairwise_restart.append(T_hat_pairwise_restart_i)
        T_hat_pairwise_restart = Transform3d(matrix=torch.stack(T_hat_pairwise_restart))
        self.update_estimation(T_hat_pairwise_restart)

    @timed
    def iterative_optimization(self, views: Tensor, T_gt: Tensor | None = None) -> None:
        """ Initialize with best SO(3) minimum from Chamfer distances between views[0] and views[1:].
            Then, iterate join generative optimization and non generative pairwise restart until no
            new poses of best losses are found.
            If `T_gt` is provided, log the rotation error after each step.
        """
        if T_gt is not None:
            R_gt = T_gt[:, :3, :3].to(views.device)
            self.errors = dict(pairwise_restarts=list(), joint_generative=list())
        self.initialize(views)
        if T_gt is not None:
            self.errors['init'] = pairwise_so3_relative_angles(R_gt, self.R_hat)  # type: ignore
        self.joint_generative_optimization()
        if T_gt is not None:
            self.errors['joint_generative'].append(pairwise_so3_relative_angles(R_gt, self.R_hat))
        old_T_hat = self.T_hat.clone()
        old_losses = self.base_method.criterion(views=self.views)
        converged = False
        iter = 1
        while not converged and iter <= self.max_iter:
            self.pairwise_non_generative_restarts()
            if T_gt is not None:
                self.errors['pairwise_restarts'].append(pairwise_so3_relative_angles(R_gt, self.R_hat))
            self.joint_generative_optimization()
            if T_gt is not None:
                self.errors['joint_generative'].append(pairwise_so3_relative_angles(R_gt, self.R_hat))
            losses = self.base_method.criterion(views=self.views)
            loss_improvements = losses <= old_losses
            old_T_hat_matrix = old_T_hat.get_matrix()
            old_R_hat = old_T_hat_matrix[:, :3, :3]
            so3_distances = so3_relative_angle(self.R_hat, old_R_hat) * 180 / torch.pi
            new_poses = so3_distances >= self.R_eps
            local_minima_escapes = torch.logical_and(loss_improvements, new_poses).sum().item()
            converged = local_minima_escapes == 0
            self.print_if_verbose((
                f'Iter n°{iter}: loss improvements={loss_improvements.sum().item()} | '
                f'New poses={new_poses.sum().item()} | Local Minima Escape={local_minima_escapes}'
            ))
            if converged:
                self.print_if_verbose((f':: BRIFF({self.base_method.__class__.__name__}) '
                                       f'converged in {iter+1} iterations.'))
                break
            # 4. Replace by new T where loss improvements
            T_hat_matrix = self.T_hat.get_matrix()
            old_T_hat_matrix[loss_improvements] = T_hat_matrix[loss_improvements]
            self.T_hat = Transform3d(matrix=old_T_hat_matrix)
            old_losses = losses.clone()
            old_T_hat = self.T_hat.clone()
            iter += 1
        if T_gt is not None:
            self.errors['final'] = pairwise_so3_relative_angles(R_gt, self.R_hat)  # type: ignore

    def __call__(self, views: Tensor, T_gt: Tensor | None = None) -> None:
        """ Once finished, runtimes can be accessed through the self.elapsed_time dictionary. """
        self.iterative_optimization(views, T_gt)
