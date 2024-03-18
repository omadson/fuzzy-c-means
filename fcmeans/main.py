from enum import Enum
from typing import Callable, Dict, Optional, Union

import numpy as np
import tqdm
from joblib import Parallel, delayed
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, validate_call


class DistanceOptions(str, Enum):
    """Implemented distances"""
    euclidean = 'euclidean'
    minkowski = 'minkowski'
    cosine = 'cosine'


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
    trained: bool = False
    n_jobs: int = Field(1, ge=1)
    verbose: Optional[bool] = False
    distance: Optional[Union[DistanceOptions, Callable]] = (
        DistanceOptions.euclidean
    )
    distance_params: Optional[Dict] = {}

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: NDArray) -> None:
        """Train the fuzzy-c-means model

        Args:
            X (NDArray): Training instances to cluster.
        """
        self.rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
        self.u = self.u / np.tile(
            self.u.sum(axis=1)[np.newaxis].T, self.n_clusters
        )
        for _ in tqdm.tqdm(
            range(self.max_iter), desc="Training", disable=not self.verbose
        ):
            u_old = self.u.copy()
            self._centers = FCM._next_centers(X, self.u, self.m)
            self.u = self.soft_predict(X)
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
        temp = FCM._dist(
            X,
            self._centers,
            self.distance,
            self.distance_params
        ) ** (2 / (self.m - 1))
        u_dist = Parallel(n_jobs=self.n_jobs)(
            delayed(
                lambda data, col: (data[:, col] / data.T).sum(0)
            )(temp, col)
            for col in range(temp.shape[1])
        )
        u_dist = np.vstack(u_dist).T
        return 1 / u_dist

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
        if self._is_trained():
            X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
            return self.soft_predict(X).argmax(axis=-1)
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    def _is_trained(self) -> bool:
        if self.trained:
            return True
        return False

    @staticmethod
    def _dist(
        A: NDArray,
        B: NDArray,
        distance: Optional[Union[DistanceOptions, Callable]] = (
            DistanceOptions.euclidean
        ),
        distance_params: Optional[Dict] = {}
    ) -> NDArray:
        """Compute the distance between two matrices"""
        if callable(distance):
            return distance(A, B, distance_params)
        elif distance == 'minkowski':
            if isinstance(distance_params, dict):
                p = distance_params.get("p", 1.0)
            else:
                p = 1.0
            return FCM._minkowski(A, B, p)
        elif distance == 'cosine':
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
        return (np.einsum("ijk->ij", (A[:, None, :] - B) ** p)) ** (1/p)

    @staticmethod
    def _cosine_similarity(A: NDArray, B: NDArray) -> NDArray:
        """Compute the cosine similarity between two matrices"""
        p1 = np.sqrt(np.sum(A**2, axis=1))[:, np.newaxis]
        p2 = np.sqrt(np.sum(B**2, axis=1))[np.newaxis, :]
        return np.dot(A, B.T) / (p1*p2)

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
        if self._is_trained():
            return self._centers
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @property
    def partition_coefficient(self) -> float:
        """Partition coefficient

        Equation 12a of
        [this paper](https://doi.org/10.1016/0098-3004(84)90020-7).
        """
        if self._is_trained():
            return np.mean(self.u**2)
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @property
    def partition_entropy_coefficient(self):
        if self._is_trained():
            return -np.mean(self.u * np.log2(self.u))
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )
