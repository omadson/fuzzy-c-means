"""Gath-Geva clustering implementation."""

import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from .gk import GK


class GG(GK):
    r"""Gath-Geva Model

    Fuzzy maximum likelihood clustering (Gath and Geva, 1989). Like GK, each
    cluster has its own covariance matrix $F_i$, and it also has a prior
    probability $\alpha_i$, so clusters can differ in shape, size and
    density. The squared distance is the inverse of a Gaussian density:

    $$\alpha_i = \frac{\sum_j u_{ij}^m}{\sum_{l,j} u_{lj}^m}
    \qquad
    d_{ij}^2 = \frac{\sqrt{\det F_i}}{\alpha_i}
    \exp\left(\tfrac{1}{2} (x_j - v_i)^\top F_i^{-1} (x_j - v_i)\right)$$

    with $F_i$ the fuzzy covariance matrix of [GK][fcmeans.GK]. Memberships
    are computed from $d_{ij}^2$ as in FCM, in the log domain so that samples
    far from every cluster do not overflow.

    The exponential distance makes GG very sensitive to its starting point,
    so the model is initialized with a GK run (same hyperparameters) and the
    GG iterations start from its partition.

    Accepts the same hyperparameters as [GK][fcmeans.GK] (`distance` cannot
    be changed).

    Attributes:
        covariances (NDArray): Covariance matrices $F_i$ with shape
        (n_clusters, n_features, n_features), set by `fit`.
        priors (NDArray): Prior probabilities $\alpha_i$, summing to one, set
        by `fit`.

    Raises:
        ReferenceError: If called without the model being trained.
        ValidationError: If `distance` is not the default.
    """

    def _init_u(self, X: NDArray) -> None:
        """Initialize `u` from a GK run."""
        gk = GK(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            m=self.m,
            error=self.error,
            random_state=self.random_state,
            init=self.init,
            reg=self.reg,
        )
        gk.fit(X)
        self.rng = gk.rng
        self.u = gk.u

    def _update_shape(self, X: NDArray) -> None:
        """Update covariances, priors and their determinant and inverse."""
        um = self.u**self.m
        self.covariances = self._covariances(X)
        self.priors = um.sum(axis=0) / um.sum()
        self._logdet = np.linalg.slogdet(self.covariances)[1]
        self._inv_cov = np.linalg.inv(self.covariances)

    def _objective(self, X: NDArray) -> float:
        """Log of the GG objective (the objective itself can overflow)."""
        diff = X[:, None, :] - self._centers
        mahalanobis = np.einsum("ncp,cpq,ncq->nc", diff, self._inv_cov, diff)
        log_d2 = 0.5 * self._logdet - np.log(self.priors) + 0.5 * mahalanobis
        with np.errstate(divide="ignore"):
            a = self.m * np.log(self.u) + log_d2
        top = a.max()
        return float(top + np.log(np.exp(a - top).sum()))

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def soft_predict(self, X: NDArray) -> NDArray:
        """Soft predict of GG

        Args:
            X (NDArray): New data to predict.

        Returns:
            NDArray: Fuzzy partition array, returned as an array with
            n_samples rows and n_clusters columns.
        """
        diff = X[:, None, :] - self._centers
        mahalanobis = np.einsum("ncp,cpq,ncq->nc", diff, self._inv_cov, diff)
        log_d2 = 0.5 * self._logdet - np.log(self.priors) + 0.5 * mahalanobis
        z = -log_d2 / (self.m - 1)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
