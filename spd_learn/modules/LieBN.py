# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause

"""Lie Group Batch Normalization for SPD matrices.

This module implements SPDBatchNormLie based on:
Ziheng Chen, Yue Song, Yunmei Liu, and Nicu Sebe,
"A Lie Group Approach to Riemannian Batch Normalization," ICLR 2024.

The implementation is integrated into ``spd_learn`` from the original LieBN
repository: https://github.com/GitZH-Chen/LieBN/tree/main/LieBN
"""

import torch

from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from ..functional import (
    airm_geodesic,
    ensure_sym,
    matrix_exp,
    matrix_inv_sqrt,
    matrix_log,
    matrix_power,
    matrix_sqrt,
)
from ..functional.batchnorm import (
    karcher_mean_iteration,
    lie_group_variance,
    spd_centering,
    spd_cholesky_congruence,
    spd_rebiasing,
)
from ..functional.metrics import (
    cholesky_exp,
    cholesky_log,
    log_euclidean_scalar_multiply,
)
from .manifold import PositiveDefiniteScalar, SymmetricPositiveDefinite


class SPDBatchNormLie(nn.Module):
    r"""Lie Group Batch Normalization for SPD matrices.

    This class implements the SPD instance of the LieBN framework, using
    the three Lie group structures on the SPD manifold, corresponding to the AIM, LEM, and LCM.

    Parameters
    ----------
    n : int
        Size of the SPD matrices (n x n).
    metric : str, default="AIM"
        Lie group invariant metric. Supported values are ``"AIM"``, ``"LEM"``,
        and ``"LCM"``.
    theta : float, default=1.0
        Power deformation parameter.
    alpha : float, default=1.0
        Frobenius norm weight in variance computation.
    beta : float, default=0.0
        Trace/logdet weight in variance computation.
    momentum : float, default=0.1
        Running statistics momentum.
    eps : float, default=1e-5
        Numerical stability constant for variance normalization.
    karcher_steps : int, default=1
        Number of Karcher flow iterations used by the AIM mean.  Iterations
        stop early when the tangent update norm falls below ``1e-5``.
    congruence : {"cholesky", "eig"}, default="cholesky"
        Implementation of the AIM congruence action (centering/biasing).
        ``"cholesky"`` uses the Cholesky factor :math:`L` of :math:`P` to
        compute :math:`L X L^T` (as in the original LieBN paper).
        ``"eig"`` uses eigendecomposition-based :math:`M^{-1/2} X M^{-1/2}`
        (matching :func:`~spd_learn.functional.spd_centering`).
        Both are mathematically equivalent; Cholesky is typically faster,
        while eigendecomposition reuses the infrastructure of
        :class:`~spd_learn.modules.SPDBatchNormMeanVar`.
        Only affects the AIM metric.
    device : torch.device or str, optional
        Device on which to create parameters and buffers.
    dtype : torch.dtype, optional
        Data type of parameters and buffers.
    """

    def __init__(
        self,
        n,
        metric="AIM",
        theta=1.0,
        alpha=1.0,
        beta=0.0,
        momentum=0.1,
        eps=1e-5,
        karcher_steps=1,
        congruence="cholesky",
        device=None,
        dtype=None,
    ):
        super().__init__()
        supported_metrics = ("AIM", "LEM", "LCM")
        if metric not in supported_metrics:
            raise ValueError(
                f"metric must be one of {supported_metrics}, got '{metric}'"
            )
        if congruence not in ("cholesky", "eig"):
            raise ValueError(
                f"congruence must be 'cholesky' or 'eig', got '{congruence}'"
            )
        self.n = n
        self.metric = metric
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.momentum = momentum
        self.eps = eps
        self.karcher_steps = karcher_steps
        self.congruence = congruence

        self.bias = nn.Parameter(torch.empty(1, n, n, device=device, dtype=dtype))
        self.shift = nn.Parameter(torch.empty((), device=device, dtype=dtype))

        if metric == "AIM":
            self.register_buffer(
                "running_mean",
                torch.eye(n, device=device, dtype=dtype).unsqueeze(0),
            )
        else:
            self.register_buffer(
                "running_mean",
                torch.zeros(1, n, n, device=device, dtype=dtype),
            )
        self.register_buffer("running_var", torch.ones((), device=device, dtype=dtype))

        self.reset_parameters()
        self._parametrize()

    @torch.no_grad()
    def reset_parameters(self):
        self.bias.zero_()
        self.bias[0].fill_diagonal_(1.0)
        self.shift.fill_(1.0)

    def _parametrize(self):
        register_parametrization(self, "bias", SymmetricPositiveDefinite())
        register_parametrization(self, "shift", PositiveDefiniteScalar())

    # ------------------------------------------------------------------
    # Thin dispatch helpers — each delegates to existing functional ops.
    # ------------------------------------------------------------------

    def _deform(self, X):
        """Map SPD matrices to the Lie algebra."""
        if self.metric == "AIM":
            return X if self.theta == 1.0 else matrix_power.apply(X, self.theta)
        if self.metric == "LEM":
            return matrix_log.apply(X)
        # LCM
        Xp = X if self.theta == 1.0 else matrix_power.apply(X, self.theta)
        return cholesky_log.apply(Xp)

    def _inv_deform(self, S):
        """Map from the Lie algebra back to SPD matrices."""
        if self.metric == "AIM":
            return S if self.theta == 1.0 else matrix_power.apply(S, 1.0 / self.theta)
        if self.metric == "LEM":
            return matrix_exp.apply(S)
        # LCM
        spd = ensure_sym(cholesky_exp.apply(S))
        return spd if self.theta == 1.0 else matrix_power.apply(spd, 1.0 / self.theta)

    def _translate(self, X, P, inverse=False):
        """Group translation (centering / biasing) in the Lie algebra."""
        if self.metric == "AIM":
            if self.congruence == "cholesky":
                return spd_cholesky_congruence(X, P, inverse=inverse)
            # Eigendecomposition path
            if inverse:
                return spd_centering(X, matrix_inv_sqrt.apply(P))
            return spd_rebiasing(X, matrix_sqrt.apply(P))
        return X - P if inverse else X + P

    def _frechet_mean(self, X_def):
        """Fréchet mean in the deformed space."""
        if self.metric == "AIM":
            batch = X_def.detach()
            mean = batch.mean(dim=0, keepdim=True)
            for _ in range(self.karcher_steps):
                mean, mean_tangent = karcher_mean_iteration(
                    batch, mean, detach=True, return_tangent=True
                )
                if mean_tangent.norm(dim=(-1, -2)).max() < 1e-5:
                    break
            return mean
        return X_def.detach().mean(dim=0, keepdim=True)

    def _scale(self, X, var):
        """Variance normalization in the Lie algebra."""
        factor = self.shift / (var + self.eps).sqrt()
        if self.metric == "AIM":
            return log_euclidean_scalar_multiply(factor, X)
        return X * factor

    # ------------------------------------------------------------------
    # Running statistics & forward
    # ------------------------------------------------------------------

    def _update_running_stats(self, batch_mean, batch_var):
        with torch.no_grad():
            if self.metric == "AIM":
                self.running_mean = airm_geodesic(
                    self.running_mean, batch_mean, self.momentum
                )
            else:
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

    def forward(self, X):
        X_def = self._deform(X)
        bias_def = self._deform(self.bias)

        if self.training:
            batch_mean = self._frechet_mean(X_def)
            X_centered = self._translate(X_def, batch_mean, inverse=True)
            if X.shape[0] > 1:
                batch_var = lie_group_variance(
                    X_centered.detach(),
                    self.metric,
                    self.alpha,
                    self.beta,
                    self.theta,
                )
                X_scaled = self._scale(X_centered, batch_var)
            else:
                batch_var = self.running_var.clone()
                X_scaled = X_centered
            self._update_running_stats(batch_mean.detach(), batch_var.detach())
        else:
            X_centered = self._translate(X_def, self.running_mean, inverse=True)
            X_scaled = self._scale(X_centered, self.running_var)

        X_biased = self._translate(X_scaled, bias_def, inverse=False)
        return self._inv_deform(X_biased)

    def extra_repr(self):
        return (
            f"n={self.n}, metric={self.metric}, theta={self.theta}, "
            f"alpha={self.alpha}, beta={self.beta}, momentum={self.momentum}, "
            f"congruence={self.congruence}"
        )
