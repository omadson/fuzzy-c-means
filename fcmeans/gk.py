"""Gustafson-Kessel clustering implementation."""

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator

from .main import FCM, DistanceOptions


class GK(FCM):
    r"""Gustafson-Kessel Model

    FCM with an adaptive distance: each cluster $i$ has its own norm matrix
    $A_i$, so clusters can be ellipsoids of different orientations instead of
    spheres (Gustafson and Kessel, 1978). With the cluster volume fixed at
    one:

    $$F_i = \frac{\sum_j u_{ij}^m (x_j - v_i)(x_j - v_i)^\top}
    {\sum_j u_{ij}^m}
    \qquad
    A_i = \det(F_i)^{1/p} F_i^{-1}
    \qquad
    d_{ij}^2 = (x_j - v_i)^\top A_i (x_j - v_i)$$

    where $p$ is the number of features. Memberships and centers are updated
    as in FCM.

    Accepts the same hyperparameters as [FCM][fcmeans.FCM], except that
    `distance` cannot be changed: the adaptive norm replaces it.

    Attributes:
        reg (float): Regularization of the covariance matrices. Each $F_i$
        gets `reg * trace(F_i) / p` added to its diagonal, which keeps it
        invertible when a cluster is degenerate (constant or collinear
        features, fewer samples than features).
        norm_matrices (NDArray): Norm matrices $A_i$ with shape
        (n_clusters, n_features, n_features), set by `fit`.

    Raises:
        ReferenceError: If called without the model being trained.
        ValidationError: If `distance` is not the default.
    """

    reg: float = Field(1e-6, ge=0.0)

    @field_validator("distance")
    @classmethod
    def _default_distance_only(cls, v):
        if v != DistanceOptions.euclidean:
            raise ValueError(
                "GK uses its own adaptive norm; `distance` is fixed."
            )
        return v

    def _update_centers(self, X: NDArray) -> None:
        """Update `_centers` and the norm matrices `A_i`."""
        super()._update_centers(X)
        diff = X[:, None, :] - self._centers
        um = self.u**self.m
        p = X.shape[1]
        F = np.einsum("nc,ncp,ncq->cpq", um, diff, diff)
        F /= um.sum(axis=0)[:, None, None]
        F += (
            self.reg * np.trace(F, axis1=1, axis2=2)[:, None, None] / p
        ) * np.eye(p)
        _, logdet = np.linalg.slogdet(F)
        scale = np.exp(logdet / p)[:, None, None]
        self.norm_matrices = scale * np.linalg.inv(F)

    def _distances(self, X: NDArray) -> NDArray:
        """Distance from each sample to each center under its own norm."""
        diff = X[:, None, :] - self._centers
        return np.sqrt(
            np.einsum("ncp,cpq,ncq->nc", diff, self.norm_matrices, diff)
        )
