""" Wrappers around the JRMPC implementation to make it usable by BRIFF. """


from __future__ import annotations
from typing import cast
from functools import partial
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from pytorch3d.transforms import Transform3d
from jrmpc import jrmpc, parallel_jrmpc_single_view_fixed_model
from .flames import chamfer_distances_between_views_and_model
from .briff_multiviews import GenerativeMultiviewsMethod


# ______________________________________ GMM Likekihood computation _______________________________________ #


sqe = lambda V, X: ((V[:, None, :] - X[None, :, :]) ** 2).sum(dim=2)


def compute_posteriors(V: Tensor, X: Tensor, S: Tensor, pk: Tensor, gamma: Tensor):
    """ V: (N, 3), X: (K, 3). Result: (N, K). """
    Q = 1 / S
    h = 2 / Q.mean()
    beta = gamma / (h * (gamma + 1))
    A = sqe(V, X)  # (Nj, K)
    A = (pk * (Q ** 1.5))[None, :] * torch.exp(-0.5 * Q[None, :] * A)  # (Nj, K)
    A = A / (A.sum(dim=1)[:, None] + beta)  # (Nj, K)
    return A


""" V: (B, N, 3), X: (K, 3). Result: (B, N, K). """
parallel_compute_posteriors = torch.vmap(compute_posteriors, in_dims=(0, None, None, None, None))


def likelihood_single_point_single_Gaussian(
    v_ij: Tensor, p_ijk: Tensor, mu_k: Tensor, sigma_k: Tensor
) -> Tensor:
    """ Compute the likelihood of the j-th point of the i-th view under
        the k-th Gaussian, parameterized by mu_k (3,) and sigma_k (scalar):
        p{i}(j,k) N(v_ijk; mu_k, sigma_k).

    Args:
        v_ij (Tensor): j-th point of the i-th view: 3D vector (3,).
        p_ijk (Tensor): mixing coefficient: probability of v_ij to belong to the k-th Gaussian.
            Scalar in [0, 1] such that sum_{k=1}^K+1 p_ijk = 1.
        mu_k (Tensor): k-th Gaussian mean, 3D vector (3,).
        sigma_k (Tensor): As Gausians are isotropic, the covariance matrix is parameterized by
            a single scalar.
    """
    cov = torch.diag(sigma_k.repeat(3))
    Gaussian = MultivariateNormal(loc=mu_k, covariance_matrix=cov, validate_args=False)
    return p_ijk * Gaussian.log_prob(v_ij).exp()


likelihood_single_point_all_Gaussians = torch.vmap(likelihood_single_point_single_Gaussian,
                                                   in_dims=(None, 0, 0, 0))

likelihood_single_point = lambda vi, pi, mu, s: likelihood_single_point_all_Gaussians(vi, pi, mu, s).sum()


likelihood_single_view_all_points = torch.vmap(likelihood_single_point, in_dims=(0, 0, None, None))
nll_single_view = lambda v_i, p_i, mu, s: - likelihood_single_view_all_points(v_i, p_i, mu, s).log().sum()


def nll_views_iterative(V_hat: list[Tensor], A: list[Tensor], model: Tensor, S: Tensor) -> Tensor:
    """ V_hat: (B, N, 3), A (B, N, K, 3), model (K, 3), S (K,). """
    return torch.hstack([nll_single_view(v_i, p_i, model, S) for v_i, p_i in zip(V_hat, A)])


# Fast version of the function above in case all point clouds are of same size
nll_views_vectorized = torch.vmap(nll_single_view, in_dims=(0, 0, None, None))


def nll_views(V_hat: Tensor | list[Tensor], model: Tensor, S: Tensor, A: Tensor | list[Tensor]) -> Tensor:
    """ V_hat: (B, N, 3), A (B, N, K, 3), model (K, 3), S (K,). """
    if isinstance(V_hat, Tensor) and isinstance(A, Tensor):
        return nll_views_vectorized(V_hat, A, model, S)
    V_hat = cast(list[Tensor], V_hat)
    A = cast(list[Tensor], A)
    return nll_views_iterative(V_hat, A, model, S)


# _________________________________________________________________________________________________________ #


def filter_model(model: Tensor, S: Tensor, A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    low_variance_clusters_mask = S < 2.5 * S.median()
    filtered_model = model[low_variance_clusters_mask]
    filtered_variances = S[low_variance_clusters_mask]
    filtered_posteriors = A[:, :, low_variance_clusters_mask]
    return filtered_model, filtered_variances, filtered_posteriors


def jrmpc_optim(
    views: Tensor, X: Tensor | None = None, gamma: float | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    if X is not None:
        # parallel pairwise non generative restarts
        R_hat, t_hat = parallel_jrmpc_single_view_fixed_model(views.mT, X.T, gamma=gamma, max_num_iter=20)
        t_hat = t_hat.squeeze(dim=2)
        model = X
        S, A = None, None
    else:
        # joint generative fit
        outputs = jrmpc(views.mT.unbind(), gamma=gamma, max_num_iter=100)
        R_hat, t_hat, model = outputs.R, outputs.t.squeeze(dim=2), outputs.X.T
        S, A = outputs.S, torch.stack(outputs.A)
    T_hat = Transform3d(device=R_hat.device).rotate(R_hat.mT).translate(t_hat)
    T_hat = T_hat.get_matrix().squeeze()
    return T_hat, model, S, A


# _________________________________________________________________________________________________________ #

class JRMPC(GenerativeMultiviewsMethod):

    """ Based JRMPC class to be inherited with various criteria in order to be used by BRIFF. """

    def __init__(self, gamma: float | None = None, filter_model: bool = True) -> None:
        self.gamma = gamma
        self.filter_model = filter_model

    def optimize(self, views: Tensor, generative: bool = False) -> Tensor:
        model = None if generative else self.model
        T_hat, self.model, S, A = jrmpc_optim(views, model, self.gamma)
        if generative:
            S, A = cast(Tensor, S), cast(Tensor, A)
            self.S, self.A = S, A
            if self.filter_model:
                self.model, self.S, self.A = filter_model(self.model, self.S, self.A)
        return T_hat


class LikelihoodJRMPC(JRMPC):

    """ JRMPC with likelihood criterion (slow). """

    def compute_priors(self) -> None:
        K = len(self.model)
        if self.gamma is not None:
            self.pk = torch.tensor(1 / (K * (self.gamma + 1)), device=self.model.device).repeat(K)
        else:  # gamma and initial_priors are both None
            self.gamma = 1 / K
            self.pk = torch.tensor(1 / (K + 1), device=self.model.device).repeat(K)
        self.gamma = cast(Tensor, self.gamma)

    def criterion(self, *, views: Tensor, model: Tensor | None = None) -> Tensor:
        if model is not None:
            self.compute_priors()
            A = parallel_compute_posteriors(views, self.model, self.S, self.gamma, self.pk)
        else:
            model = self.model
            A = self.A
        model = self.model if model is None else model
        return nll_views(views, model, self.S, A)


class ChamferJRMPC(JRMPC):

    """ JRMPC with Chamfer criterion (fast). """

    def __init__(
        self,
        gamma: float | None = None,
        filter_model: bool = True,
        from_model: bool = True,
        bidirectional: bool = False,
        trim_ratio: float = 0.
    ) -> None:
        self.gamma = gamma
        self.filter_model = filter_model
        self.chamfer_distances = partial(chamfer_distances_between_views_and_model,
                                         from_model=from_model,
                                         bidirectional=bidirectional,
                                         trim_ratio=trim_ratio)

    def criterion(self, *, views: Tensor, model: Tensor | None = None) -> Tensor:
        model = self.model if model is None else model
        return self.chamfer_distances(views=views, model=model)
