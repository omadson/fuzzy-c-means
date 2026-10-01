"""Fuzzy C-means clustering implementation."""

from enum import Enum
from typing import Callable, Literal, Optional, Union

import numpy as np
import tqdm
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, validate_call


class DistanceOptions(str, Enum):
    """Implemented distances"""

    euclidean = "euclidean"
    minkowski = "minkowski"
    cosine = "cosine"


class FCM(BaseModel):
    r"""Fuzzy C-means Model

    Attributes:
        n_clusters (int): The number of clusters to form as well as the number
        of centroids to generate by the fuzzy C-means.
        max_iter (int): Maximum number of iterations of the fuzzy C-means
        algorithm for a single run.
        m (float): Degree of fuzziness: $m \in (1, \infty)$.
        error (float): Relative tolerance with regards to Frobenius norm of
        the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        random_state (Optional[int]): Determines random number generation for
        centroid initialization.
        Use an int to make the randomness deterministic.
        init (str): Initialization of the partition. `"random"` draws a
        random fuzzy partition; `"k-means++"` seeds the centers with the
        k-means++ rule (each new center is drawn with probability
        proportional to its squared distance to the closest chosen one) and
        derives the partition from them. Ignored by `FCMedoids`, which
        always seeds with k-means++.
        trained (bool): Variable to store whether or not the model has been
        trained.

    Returns:
        FCM: A FCM model.

    Raises:
        ReferenceError: If called without the model being trained
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    n_clusters: int = Field(5, ge=1)
    max_iter: int = Field(150, ge=1, le=1000)
    m: float = Field(2.0, ge=1.0)
    error: float = Field(1e-5, ge=1e-9)
    random_state: Optional[int] = None
    init: Literal["random", "k-means++"] = "random"
    trained: bool = False
    verbose: Optional[bool] = False
    distance: Optional[Union[DistanceOptions, Callable]] = (
        DistanceOptions.euclidean
    )
    distance_params: Optional[dict] = {}

    def _init_u(self, X: NDArray) -> None:
        """Initialize the fuzzy partition matrix `u`."""
        self.rng = np.random.default_rng(self.random_state)
        if self.init == "k-means++":
            self._centers = self._seed_centers(X)
            self.u = FCM._memberships(
                FCM._dist(
                    X, self._centers, self.distance, self.distance_params
                ),
                self.m,
            )
            return
        u = self.rng.uniform(size=(X.shape[0], self.n_clusters))
        self.u = u / u.sum(axis=1, keepdims=True)

    def _seed_centers(self, X: NDArray) -> NDArray:
        """Pick `n_clusters` samples with the k-means++ rule."""
        n = X.shape[0]
        chosen = [self.rng.integers(n)]
        closest = self._sq_dist_to(X, chosen[0])
        for _ in range(1, self.n_clusters):
            total = closest.sum()
            chosen.append(
                self.rng.choice(n, p=closest / total if total > 0 else None)
            )
            closest = np.minimum(closest, self._sq_dist_to(X, chosen[-1]))
        return X[chosen]

    def _sq_dist_to(self, X: NDArray, i: int) -> NDArray:
        """Squared distance from every sample to sample `i`."""
        d = FCM._dist(X, X[[i]], self.distance, self.distance_params)
        return d[:, 0] ** 2

    def _update_centers(self, X: NDArray) -> None:
        """Update `_centers` from the current partition matrix `u`."""
        self._centers = FCM._next_centers(X, self.u, self.m)

    def _distances(self, X: NDArray) -> NDArray:
        """Distance from each sample in X to each center."""
        return FCM._dist(X, self._centers, self.distance, self.distance_params)

    def _update_u(self, X: NDArray) -> None:
        """Update `u` from the current centers."""
        self.u = self.soft_predict(X)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: NDArray) -> None:
        """Train the fuzzy-c-means model

        Args:
            X (NDArray): Training instances to cluster.
        """
        self._init_u(X)
        for _ in tqdm.tqdm(
            range(self.max_iter), desc="Training", disable=not self.verbose
        ):
            u_old = self.u.copy()
            self._update_centers(X)
            self._update_u(X)
            # Stopping rule
            if np.linalg.norm(self.u - u_old) < self.error:
                break
        self.trained = True

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def soft_predict(self, X: NDArray) -> NDArray:
        """Soft predict of FCM

        Args:
            X (NDArray): New data to predict.

        Returns:
            NDArray: Fuzzy partition array, returned as an array with
            n_samples rows and n_clusters columns.
        """
        return FCM._memberships(self._distances(X), self.m)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: NDArray) -> NDArray:
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X (NDArray): New data to predict.

        Raises:
            ReferenceError: If it called without the model being trained.

        Returns:
            NDArray: Index of the cluster each sample belongs to.
        """
        self._require_trained()
        X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
        return self.soft_predict(X).argmax(axis=-1)

    def _require_trained(self) -> None:
        if not self.trained:
            raise ReferenceError(
                "You need to train the model. Run `.fit()` method to this."
            )

    @staticmethod
    def _memberships(d: NDArray, m: float) -> NDArray:
        """Fuzzy partition from the sample-to-center distances `d`."""
        with np.errstate(divide="ignore", invalid="ignore"):
            temp = d ** (2 / (m - 1))
            u = 1.0 / (temp * (1.0 / temp).sum(axis=1, keepdims=True))
        # a sample on a center belongs only to it (evenly split on ties)
        zero = d == 0
        rows = zero.any(axis=1)
        u[rows] = zero[rows] / zero[rows].sum(axis=1, keepdims=True)
        return u

    @staticmethod
    def _dist(
        A: NDArray,
        B: NDArray,
        distance: Optional[Union[DistanceOptions, Callable]] = (
            DistanceOptions.euclidean
        ),
        distance_params: Optional[dict] = {},
    ) -> NDArray:
        """Compute the distance between two matrices"""
        if callable(distance):
            return distance(A, B, distance_params)
        elif distance == "minkowski":
            if isinstance(distance_params, dict):
                p = distance_params.get("p", 1.0)
            else:
                p = 1.0
            return FCM._minkowski(A, B, p)
        elif distance == "cosine":
            return FCM._cosine(A, B)
        else:
            return FCM._euclidean(A, B)

    @staticmethod
    def _euclidean(A: NDArray, B: NDArray) -> NDArray:
        """Compute the euclidean distance between two matrices"""
        return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))

    @staticmethod
    def _minkowski(A: NDArray, B: NDArray, p: float) -> NDArray:
        """Compute the minkowski distance between two matrices"""
        return np.einsum("ijk->ij", np.abs(A[:, None, :] - B) ** p) ** (1 / p)

    @staticmethod
    def _cosine_similarity(A: NDArray, B: NDArray) -> NDArray:
        """Compute the cosine similarity between two matrices"""
        p1 = np.sqrt(np.sum(A**2, axis=1))[:, np.newaxis]
        p2 = np.sqrt(np.sum(B**2, axis=1))[np.newaxis, :]
        return np.dot(A, B.T) / (p1 * p2)

    @staticmethod
    def _cosine(A: NDArray, B: NDArray) -> NDArray:
        """Compute the cosine distance between two matrices"""
        return np.abs(1 - FCM._cosine_similarity(A, B))

    @staticmethod
    def _next_centers(X: NDArray, u: NDArray, m: float):
        """Update cluster centers"""
        um = u**m
        return (X.T @ um / np.sum(um, axis=0)).T

    @property
    def centers(self) -> NDArray:
        self._require_trained()
        return self._centers

    @property
    def partition_coefficient(self) -> float:
        """Partition coefficient

        Equation 12a of
        [this paper](https://doi.org/10.1016/0098-3004(84)90020-7).
        """
        self._require_trained()
        return np.sum(np.mean(self.u**2, axis=0))

    @property
    def partition_entropy_coefficient(self):
        self._require_trained()
        return -np.sum(np.mean(self.u * np.log2(self.u), axis=0))
