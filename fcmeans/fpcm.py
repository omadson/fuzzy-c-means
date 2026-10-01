"""Fuzzy-possibilistic C-means clustering implementation."""

from numpy.typing import NDArray
from pydantic import Field

from .main import FCM


class FPCM(FCM):
    r"""Fuzzy-possibilistic C-means Model

    Mixed model of Pal, Pal and Bezdek (1997). Each sample has a fuzzy
    membership $u_{ij}$ (as in FCM, summing to one over the clusters) and a
    typicality $t_{ij}$ (summing to one over the samples of each cluster).
    Centers are computed from both:

    $$v_i = \frac{\sum_j (u_{ij}^m + t_{ij}^\eta) x_j}
    {\sum_j (u_{ij}^m + t_{ij}^\eta)}$$

    Accepts the same hyperparameters as [FCM][fcmeans.FCM], plus `eta`.
    `soft_predict` and `predict` use the fuzzy memberships only, since
    typicality is normalized over the training samples and is not defined
    for new data.

    Attributes:
        eta (float): Typicality exponent: $\eta \in (1, \infty)$.
        t (NDArray): Typicality matrix of the training data, with n_samples
        rows and n_clusters columns, set by `fit`.

    Raises:
        ReferenceError: If called without the model being trained.
    """

    eta: float = Field(2.0, gt=1.0)

    def _init_u(self, X: NDArray) -> None:
        """Initialize `u` randomly and derive `t` from it."""
        super()._init_u(X)
        self.t = self.u / self.u.sum(axis=0, keepdims=True)

    def _objective(self, X: NDArray) -> float:
        """Fuzzy and typicality weighted sum of squared distances."""
        d = FCM._dist(X, self._centers, self.distance, self.distance_params)
        return float(((self.u**self.m + self.t**self.eta) * d**2).sum())

    def _update_centers(self, X: NDArray) -> None:
        """Update `_centers` from the fuzzy and typicality weights."""
        w = self.u**self.m + self.t**self.eta
        self._centers = (X.T @ w / w.sum(axis=0)).T

    def _update_u(self, X: NDArray) -> None:
        """Update `u` and `t` from the current centers."""
        super()._update_u(X)
        d = FCM._dist(X, self._centers, self.distance, self.distance_params)
        w = d ** (-2 / (self.eta - 1))
        self.t = w / w.sum(axis=0, keepdims=True)
