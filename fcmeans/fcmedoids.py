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

    The distance between all pairs of training samples is computed once, so
    `fit` needs O(n²) memory.

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
        # ponytail: full n x n matrix, O(n^2) memory. Chunk or restrict
        # candidate medoids if it matters.
        chunks = np.array_split(X, max(1, n // 256))
        self._d2 = np.vstack(
            [
                FCM._dist(c, X, self.distance, self.distance_params) ** 2
                for c in chunks
            ]
        )
        # k-means++ seeding on the distance matrix: each new medoid is drawn
        # with probability proportional to its cost to the closest chosen one
        chosen = [self.rng.integers(n)]
        closest = self._d2[:, chosen[0]]
        for _ in range(1, self.n_clusters):
            total = closest.sum()
            chosen.append(
                self.rng.choice(n, p=closest / total if total > 0 else None)
            )
            closest = np.minimum(closest, self._d2[:, chosen[-1]])
        self.medoid_indices = np.array(chosen)
        self._centers = X[self.medoid_indices]
        self.u = self.soft_predict(X)

    def _update_centers(self, X: NDArray) -> None:
        """Update the medoids from the current partition matrix `u`."""
        cost = (self.u**self.m).T @ self._d2
        self.medoid_indices = cost.argmin(axis=1)
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
