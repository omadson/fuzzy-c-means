"""Fuzzy C-medoids clustering implementation."""

import numpy as np
from numpy.typing import NDArray
from pydantic import validate_call

from .main import FCM


class FCMedoids(FCM):
    r"""Fuzzy C-medoids Model

    FCM whose prototypes are restricted to be samples of the data set, the
    *medoids* (Krishnapuram, Joshi, Nasraoui and Yi, 2001). It minimizes the
    FCM objective with $v_i \in X$:

    $$J = \sum_{i=1}^{c} \sum_{j=1}^{n} u_{ij}^m d^2(x_j, v_i)$$

    Memberships are updated as in FCM. Each medoid is the sample that
    minimizes the membership-weighted cost of its cluster:

    $$v_i = \operatorname*{arg\,min}_{x_k \in X}
    \sum_{j=1}^{n} u_{ij}^m d^2(x_j, x_k)$$

    Since the search only compares distances, any `distance` of
    [FCM][fcmeans.FCM] works, including a custom callable. The initial
    medoids are seeded with k-means++ on the distance matrix, using
    `random_state`, and the model has converged when the medoids stop
    changing.

    With the default euclidean distance each medoid is the sample closest to
    the membership-weighted mean of its cluster, so `fit` needs O(n) memory.
    Any other distance computes the distances between all pairs of training
    samples once, which needs O(n²) memory.

    Accepts the same hyperparameters as [FCM][fcmeans.FCM].

    Attributes:
        medoid_indices (NDArray): Indices in the training data of the
        medoids, set by `fit`. `centers` is the training data at these
        indices.

    Raises:
        ReferenceError: If called without the model being trained.
        ValueError: If `n_clusters` is greater than the number of samples.
    """

    def _init_u(self, X: NDArray) -> None:
        """Draw the initial medoids and derive `u` from them."""
        n = X.shape[0]
        if self.n_clusters > n:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot exceed the number "
                f"of samples ({n})."
            )
        self.rng = np.random.default_rng(self.random_state)
        # squared euclidean has a closed form (see `_update_centers`); any
        # other distance needs all pairwise costs.
        # ponytail: full n x n matrix, O(n^2) memory. Chunk or restrict
        # candidate medoids if it matters.
        self._closed_form = self.distance == "euclidean"
        if not self._closed_form:
            self._d2 = np.empty((n, n))
            for start in range(0, n, 256):
                rows = slice(start, start + 256)
                self._d2[rows] = (
                    FCM._dist(X[rows], X, self.distance, self.distance_params)
                    ** 2
                )
        chosen = self._seed_indices(X)
        self.medoid_indices = np.array(chosen)
        self._centers = X[self.medoid_indices]
        self.u = self.soft_predict(X)

    def _update_centers(self, X: NDArray) -> None:
        """Update the medoids from the current partition matrix `u`."""
        um = self.u**self.m
        if self._closed_form:
            # sum_j w_j |x_j - x_k|^2 = |x_k - mean_w|^2 * sum(w) + const
            means = (X.T @ um / um.sum(axis=0)).T
            self.medoid_indices = FCM._sq_euclidean(X, means).argmin(axis=0)
        else:
            self.medoid_indices = (um.T @ self._d2).argmin(axis=1)
        self._centers = X[self.medoid_indices]

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: NDArray) -> None:
        """Train the fuzzy c-medoids model

        Args:
            X (NDArray): Training instances to cluster.
        """
        try:
            super().fit(X)
        finally:
            self.__dict__.pop("_d2", None)
