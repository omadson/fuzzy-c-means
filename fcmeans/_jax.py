from jax import jit
from jax import random
from jax import numpy as np
import time
import logging
logging.disable(logging.WARNING)


class FCM:
    """Fuzzy C-means
    Parameters
    ----------
    n_clusters: int, optional (default=10)
        The number of clusters to form as well as the number of
        centroids to generate

    max_iter: int, optional (default=150)
        Hard limit on iterations within solver.

    m: float, optional (default=2.0)
        Exponent for the fuzzy partition matrix, specified as a
        scalar greater than 1.0. This option controls the amount of
        fuzzy overlap between clusters, with larger values indicating
        a greater degree of overlap.


    error: float, optional (default=1e-5)
        Tolerance for stopping criterion.

    random_state: int, optional (default=42)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.

    Attributes
    ----------
    n_samples: int
        Number of examples in the data set

    n_features: int
        Number of features in samples of the data set

    u: array, shape = [n_samples, n_clusters]
        Fuzzy partition array, returned as an array with n_samples rows
        and n_clusters columns. Element u[i,j] indicates the degree of
        membership of the jth data point in the ith cluster. For a given
        data point, the sum of the membership values for all clusters is one.

    centers: array, shape = [n_class-1, n_SV]
        Final cluster centers, returned as an array with n_clusters rows
        containing the coordinates of each cluster center. The number of
        columns in centers is equal to the dimensionality of the data being
        clustered.

    Methods
    -------
    fit(X)
        fit the data

    _predict(X)
        use fitted model and output cluster memberships

    predict(X)
        use fitted model and output 1 cluster for each sample

    References
    ----------
    .. [1] `Pattern Recognition with Fuzzy Objective Function Algorithms
        <https://doi.org/10.1007/978-1-4757-0450-1>`_
    .. [2] `FCM: The fuzzy c-means clustering algorithm
        <https://doi.org/10.1016/0098-3004(84)90020-7>`_

    """

    def __init__(
        self, n_clusters=10, max_iter=150, m=2, error=1e-5, random_state=42
    ):
        assert m > 1
        self.u, self.centers = None, None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        if not random_state:
            self.key = random.PRNGKey(int(time.time()))
        else:
            self.key = random.PRNGKey(random_state)

    def fit(self, X):
        """Compute fuzzy C-means clustering.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training instances to cluster.
        """
        self.n_samples = X.shape[0]
        self.u = random.uniform(key=self.key, shape=(
            self.n_samples, self.n_clusters))
        self.u = self.u / np.tile(self.u.sum(axis=1)
                                  [np.newaxis].T, self.n_clusters)
        for iteration in range(self.max_iter):
            u_old = self.u.copy()
            self.centers = FCM._next_centers(X, self.u, self.m)
            self.u = self.__predict(X)
            # Stopping rule
            if np.linalg.norm(self.u - u_old) < self.error:
                break

    def __predict(self, X):
        """
        Parameters
        ----------
        X : array, shape = [n_samples, n_features]
            New data to predict.

        Returns
        -------
        u: array, shape = [n_samples, n_clusters]
            Fuzzy partition array, returned as an array with n_samples rows
            and n_clusters columns.
        """
        temp = FCM._dist(X, self.centers) ** float(2 / (self.m - 1))
        denominator_ = temp.reshape(
            (X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
        denominator_ = temp[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)

    def predict(self, X):
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
        X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
        return self.__predict(X).argmax(axis=-1)

    @staticmethod
    @jit
    def _dist(A, B):
        """Compute the euclidean distance two matrices"""
        return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2))

    @staticmethod
    @jit
    def _next_centers(X, u, m):
        """Update cluster centers"""
        um = u ** m
        return (X.T @ um / np.sum(um, axis=0)).T

    # partition coefficient (Equation 12a of https://doi.org/10.1016/0098-3004(84)90020-7)
    @property
    def partition_coefficient(self):
        if hasattr(self, "u"):
            return np.sum(self.u ** 2) / self.n_samples
        else:
            raise ReferenceError(
                "You need to train the model first. You can use `.fit()` "
                "method to this."
            )

    @property
    def partition_entropy_coefficient(self):
        if hasattr(self, "u"):
            return -np.sum(self.u * np.log2(self.u)) / self.n_samples
        else:
            raise ReferenceError(
                "You need to train the model first. You can use `.fit()` "
                "method to this."
            )
