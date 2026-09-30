"""Kernel fuzzy C-means clustering implementation."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator

from .main import FCM, DistanceOptions


class KFCM(FCM):
    r"""Kernel Fuzzy C-means Model

    FCM with the distance measured in the feature space of a Gaussian kernel
    $K(x, v) = \exp(-\gamma \|x - v\|^2)$, while the prototypes stay in the
    input space (Wu, Xie and Yu, 2003):

    $$d^2(x, v) = \|\phi(x) - \phi(v)\|^2 = 2 (1 - K(x, v))$$

    Memberships are updated as in FCM with this distance. Centers are the
    kernel-weighted means

    $$v_i = \frac{\sum_j u_{ij}^m K(x_j, v_i) x_j}
    {\sum_j u_{ij}^m K(x_j, v_i)}$$

    so samples far from a center, such as outliers, get almost no weight.

    Accepts the same hyperparameters as [FCM][fcmeans.FCM], plus `gamma`;
    `distance` cannot be changed.

    Attributes:
        gamma (Optional[float]): Kernel coefficient, greater than zero. If
        None, `1 / (n_features * X.var())` is used.
        gamma_ (float): Value of `gamma` used by the fitted model.

    Raises:
        ReferenceError: If called without the model being trained.
        ValidationError: If `distance` is not the default or `gamma` is not
        positive.
    """

    gamma: Optional[float] = Field(None, gt=0.0)

    @field_validator("distance")
    @classmethod
    def _default_distance_only(cls, v):
        if v != DistanceOptions.euclidean:
            raise ValueError(
                "KFCM uses the Gaussian kernel distance; `distance` is fixed."
            )
        return v

    def _init_u(self, X: NDArray) -> None:
        """Initialize `u`, the centers and `gamma_` from a FCM run."""
        fcm = FCM(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            m=self.m,
            error=self.error,
            random_state=self.random_state,
        )
        fcm.fit(X)
        self.rng = fcm.rng
        self.u = fcm.u
        self._centers = fcm.centers
        var = X.var()
        self.gamma_ = self.gamma or (
            1.0 / (X.shape[1] * var) if var > 0 else 1.0
        )

    def _kernel(self, X: NDArray) -> NDArray:
        """Gaussian kernel between the samples and the centers."""
        return np.exp(-self.gamma_ * self._sq_dist(X))

    def _sq_dist(self, X: NDArray) -> NDArray:
        """Squared euclidean distance between samples and centers."""
        return np.einsum("ijk->ij", (X[:, None, :] - self._centers) ** 2)

    def _update_centers(self, X: NDArray) -> None:
        """Update `_centers` as kernel-weighted means."""
        w = self.u**self.m * self._kernel(X)
        den = w.sum(axis=0)[:, None]
        # a center that no sample reaches (all weights 0) stays where it is
        self._centers = np.divide(
            (X.T @ w).T, den, out=self._centers.copy(), where=den > 0
        )

    def _distances(self, X: NDArray) -> NDArray:
        """Distance in feature space, from `1 - K` computed without loss."""
        return np.sqrt(-2.0 * np.expm1(-self.gamma_ * self._sq_dist(X)))
