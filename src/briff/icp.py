""" Drop-in replacement for Pytorch3D's iterative_closest_point function with trimming and patience. """

from typing import Optional, TYPE_CHECKING, Union, cast

import torch
from pytorch3d.ops import knn_points
from pytorch3d.ops import utils as oputil
from pytorch3d.ops.points_alignment import (
    SimilarityTransform, ICPSolution, corresponding_points_alignment, _apply_similarity_transform
)

if TYPE_CHECKING:
    from pytorch3d.structures.pointclouds import Pointclouds


def iterative_closest_point(
    X: Union[torch.Tensor, "Pointclouds"],
    Y: Union[torch.Tensor, "Pointclouds"],
    init_transform: Optional[SimilarityTransform] = None,
    trim_ratio: Union[float, torch.Tensor] = 0.,
    max_iterations: int = 100,
    relative_rmse_thr: float = 1e-6,
    patience: int = 2,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    verbose: bool = False,
) -> ICPSolution:
    """
    Executes the iterative closest point (ICP) algorithm [1, 2] in order to find
    a similarity transformation (rotation `R`, translation `T`, and
    optionally scale `s`) between two given differently-sized sets of
    `d`-dimensional points `X` and `Y`, such that:

    `s[i] X[i] R[i] + T[i] = Y[NN[i]]`,

    for all batch indices `i` in the least squares sense. Here, Y[NN[i]] stands
    for the indices of nearest neighbors from `Y` to each point in `X`.
    Note, however, that the solution is only a local optimum.

    Args:
        **X**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_X, d)` or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_Y, d)` or a `Pointclouds` object.
        **init_transform**: A named-tuple `SimilarityTransform` of tensors
            `R`, `T, `s`, where `R` is a batch of orthonormal matrices of
            shape `(minibatch, d, d)`, `T` is a batch of translations
            of shape `(minibatch, d)` and `s` is a batch of scaling factors
            of shape `(minibatch,)`.
        **trim_ratio**: A float or 1d `Tensor` of shape `(minibatch,)` in [0, 1]
            specifying the ratio of outliers in each point cloud. If float, assume
            the same outliers ratio for all point clouds in the batch. Outliers will
            be detected by taking the `trim_ratio * num_points_X` highest values of
            `s[i] X[i] R[i] + T[i] = Y[NN[i]]`.
        **max_iterations**: The maximum number of ICP iterations.
        **relative_rmse_thr**: A threshold on the relative root mean squared error
            used to terminate the algorithm.
        **patience**: Number of consecutive steps during which `relative_rmse` must be
            below `relative_rmse_thr` to terminate the algorithm. This is required because
            during convergence, the relative RMSE may be negative for a few steps.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes the identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **verbose**: If `True`, prints status messages during each ICP iteration.

    Returns:
        A named tuple `ICPSolution` with the following fields:
        **converged**: A boolean flag denoting whether the algorithm converged
            successfully (=`True`) or not (=`False`).
        **rmse**: Attained root mean squared error after termination of ICP.
        **Xt**: The point cloud `X` transformed with the final transformation
            (`R`, `T`, `s`). If `X` is a `Pointclouds` object, returns an
            instance of `Pointclouds`, otherwise returns `torch.Tensor`.
        **RTs**: A named tuple `SimilarityTransform` containing
        a batch of similarity transforms with fields:
            **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
            **T**: Batch of translations of shape `(minibatch, d)`.
            **s**: batch of scaling factors of shape `(minibatch, )`.
        **t_history**: A list of named tuples `SimilarityTransform`
            the transformation parameters after each ICP iteration.

    References:
        [1] Besl & McKay: A Method for Registration of 3-D Shapes. TPAMI, 1992.
        [2] https://en.wikipedia.org/wiki/Iterative_closest_point
    """

    # make sure we convert input Pointclouds structures to
    # padded tensors of shape (N, P, 3)
    Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
    Yt, num_points_Y = oputil.convert_pointclouds_to_tensor(Y)

    Xt = cast(torch.Tensor, Xt)
    Yt = cast(torch.Tensor, Yt)

    b, size_X, dim = Xt.shape

    if (Xt.shape[2] != Yt.shape[2]) or (Xt.shape[0] != Yt.shape[0]):
        raise ValueError(
            "Point sets X and Y have to have the same "
            + "number of batches and data dimensions."
        )

    if ((num_points_Y < Yt.shape[1]).any() or (num_points_X < Xt.shape[1]).any()) and (
        num_points_Y != num_points_X
    ).any():
        # we have a heterogeneous input (e.g. because X/Y is
        # an instance of Pointclouds)
        mask_X = (
            torch.arange(size_X, dtype=torch.int64, device=Xt.device)[None]
            < num_points_X[:, None]
        ).type_as(Xt)
    else:
        mask_X = Xt.new_ones(b, size_X)

    # clone the initial point cloud
    Xt_init = Xt.clone()

    if init_transform is not None:
        # parse the initial transform from the input and apply to Xt
        try:
            R, T, s = init_transform
            assert (
                R.shape == torch.Size((b, dim, dim))
                and T.shape == torch.Size((b, dim))
                and s.shape == torch.Size((b,))
            )
        except Exception:
            raise ValueError(
                "The initial transformation init_transform has to be "
                "a named tuple SimilarityTransform with elements (R, T, s). "
                "R are dim x dim orthonormal matrices of shape "
                "(minibatch, dim, dim), T is a batch of dim-dimensional "
                "translations of shape (minibatch, dim) and s is a batch "
                "of scalars of shape (minibatch,)."
            ) from None
        # apply the init transform to the input point cloud
        Xt = _apply_similarity_transform(Xt, R, T, s)
    else:
        # initialize the transformation with identity
        R = oputil.eyes(dim, b, device=Xt.device, dtype=Xt.dtype)
        T = Xt.new_zeros((b, dim))
        s = Xt.new_ones(b)

    # initialize trim ratio parameter
    if isinstance(trim_ratio, (float, int)):
        trim_ratio = torch.as_tensor(trim_ratio)
    trim_ratio = trim_ratio.to(Xt.device)  # type: ignore
    if trim_ratio.ndim == 0:
        trim_ratio = trim_ratio.repeat(b)
    quantiles = 1 - trim_ratio
    trim = trim_ratio.min() > 0.0

    # initial mask: no trim considered, only padding
    mask = mask_X.bool().clone()

    prev_rmse = None
    rmse = None
    iteration = -1
    converged = False
    no_improvements = 0

    # initialize the transformation history
    t_history = []

    # the main loop over ICP iterations
    for iteration in range(max_iterations):
        Xt_nn_points = knn_points(
            Xt, Yt, lengths1=num_points_X, lengths2=num_points_Y, K=1, return_nn=True
        ).knn[:, :, 0, :]

        # get the alignment of the nearest neighbors from Yt with Xt_init
        R, T, s = corresponding_points_alignment(
            Xt_init,
            Xt_nn_points,
            weights=mask,
            estimate_scale=estimate_scale,
            allow_reflection=allow_reflection,
        )

        # apply the estimated similarity transform to Xt_init
        Xt = _apply_similarity_transform(Xt_init, R, T, s)

        # add the current transformation to the history
        t_history.append(SimilarityTransform(R, T, s))

        # compute the root mean squared error
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        Xt_sq_diff = ((Xt - Xt_nn_points) ** 2).sum(2)

        # trimming: select `1 - trim_ratio` lowest distances.
        if trim:
            # I think we can't avoid looping because of the padding mask. But it gets a lot slower so I check
            # if we can ignore the mask (homogeneous inputs) to be faster.
            if len(num_points_X.unique()) > 1:
                diff_thresholds = torch.stack([x[m].quantile(q)
                                               for x, m, q in zip(Xt_sq_diff, mask_X.bool(), quantiles)])
            else:
                Xt_sq_diff_sorted, _ = Xt_sq_diff.sort(dim=1)
                N = num_points_X[0]  # all pointclouds in X have the same number of points
                quantile_indices = (quantiles * (N - 1)).long()
                diff_thresholds = Xt_sq_diff_sorted[torch.arange(b), quantile_indices]
            # diff_thresholds = Xt_sq_diff[mask_X.bool()].quantile(1 - trim_ratio)
            mask_trim = Xt_sq_diff < diff_thresholds[:, None]
            # final mask is (trim_mask AND pad_mask)
            mask = torch.logical_and(mask_trim, mask_X)

        rmse = oputil.wmean(Xt_sq_diff[:, :, None], mask).sqrt()[:, 0, 0]

        # compute the relative rmse
        if prev_rmse is None:
            relative_rmse = rmse.new_ones(b)
        else:
            relative_rmse = (prev_rmse - rmse) / prev_rmse

        if verbose:
            rmse_msg = (
                f"ICP iteration {iteration}: mean/max rmse = "
                + f"{rmse.mean():1.2e}/{rmse.max():1.2e} "
                + f"; mean relative rmse = {relative_rmse.mean():1.2e}"
            )
            print(rmse_msg)

        # check for convergence: we need a patience because relative RMSE could be negative
        # for a few steps while converging
        if (relative_rmse <= relative_rmse_thr).all():
            no_improvements += 1
        else:
            no_improvements = 0

        if no_improvements == patience:
            converged = True
            break

        # update the previous rmse
        prev_rmse = rmse

    if verbose:
        if converged:
            print(f"ICP has converged in {iteration + 1} iterations.")
        else:
            print(f"ICP has not converged in {max_iterations} iterations.")

    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(converged, rmse, Xt, SimilarityTransform(R, T, s), t_history)  # type: ignore


def icp(sources: torch.Tensor, models: torch.Tensor,
        batch_size: Union[int, float] = 2e4, **kwargs) -> torch.Tensor:
    """ Nothing more than a wrapper around the `iterative_closest_point()` function handling batching
        and returning only the estimated rigid motions. Ready to use with FLAMES.
    """
    bs = int(batch_size)
    results = [iterative_closest_point(s, m, **kwargs) for s, m in zip(sources.split(bs), models.split(bs))]
    R = torch.vstack([r.RTs.R for r in results])
    t = torch.vstack([r.RTs.T for r in results])
    T = oputil.eyes(4, len(sources), device=sources.device)
    T[:, :3, :3], T[:, 3, :3] = R, t
    return T
