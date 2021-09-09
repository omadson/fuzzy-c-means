from typing import Optional

import numpy as np
from pydantic import BaseModel, Extra, Field, validate_arguments

from .my_typing import Array


class FCM(BaseModel):
    n_clusters: int = Field(5, ge=1, le=100)
    max_iter: int = Field(150, ge=1, le=1000)
    m: float = Field(2.0, ge=1.0)
    error: float = Field(1e-5, ge=1e-9)
    random_state: Optional[int] = None
    trained: bool = Field(False, const=True)

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

    @validate_arguments
    def fit(self, X: Array[float]) -> None:
        """Train the fuzzy-c-means model..

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training instances to cluster.
        """
        self.rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
        self.u = self.u / np.tile(self.u.sum(axis=1)
                                  [np.newaxis].T, self.n_clusters)
        for _ in range(self.max_iter):
            u_old = self.u.copy()
            self._centers = FCM._next_centers(X, self.u, self.m)
            self.u = self.soft_predict(X)
            # Stopping rule
            if np.linalg.norm(self.u - u_old) < self.error:
                break
        self.trained = True

    def soft_predict(self, X: Array[float]) -> Array[float]:
        """Soft predict of FCM 

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        array, shape = [n_samples, n_clusters]
            Fuzzy partition array, returned as an array with n_samples rows
            and n_clusters columns.
        """
        temp = FCM._dist(X, self._centers) ** float(2 / (self.m - 1))
        denominator_ = temp.reshape(
            (X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)

    @validate_arguments
    def predict(self, X: Array[float]):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape = [n_samples,]
            Index of the cluster each sample belongs to.
        """
        if self.is_trained():
            X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
            return self.soft_predict(X).argmax(axis=-1)

    def is_trained(self) -> None:
        if self.trained:
            return True
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @staticmethod
    def _dist(A, B):
        """Compute the euclidean distance two matrices"""
        return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))

    @staticmethod
    def _next_centers(X, u, m):
        """Update cluster centers"""
        um = u ** m
        return (X.T @ um / np.sum(um, axis=0)).T

    @property
    def centers(self):
        if self.is_trained():
            return self._centers

    @property
    def partition_coefficient(self) -> float:
        """Partition coefficient (Equation 12a of https://doi.org/10.1016/0098-3004(84)90020-7)

        Returns
        -------
        float
            partition coefficient of clustering model
        """
        if self.is_trained():
            return np.mean(self.u ** 2)

    @property
    def partition_entropy_coefficient(self):
        if self.is_trained():
            return -np.mean(self.u * np.log2(self.u))
