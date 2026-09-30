"""Possibilistic C-means clustering implementation."""

from numpy.typing import NDArray
from pydantic import validate_call

from .main import FCM


class PCM(FCM):
    r"""Possibilistic C-means Model

    Relaxes the FCM constraint $\sum_i u_{ij} = 1$, so $u_{ij}$ is the
    *typicality* of sample $j$ with respect to cluster $i$ rather than a
    share of membership. Outliers get low typicality in every cluster.

    The model is initialized with a FCM run: its partition gives the
    starting centers and the scale parameters $\eta_i$ (Krishnapuram and
    Keller, 1993), which are kept fixed afterwards.

    Membership update:

    $$u_{ij} = \frac{1}{1 + (d_{ij}^2 / \eta_i)^{1/(m-1)}}$$

    Accepts the same hyperparameters as [FCM][fcmeans.FCM].

    Attributes:
        eta (NDArray): Scale parameter of each cluster, set by `fit`.

    Raises:
        ReferenceError: If called without the model being trained.
        NotImplementedError: If `partition_coefficient` or
        `partition_entropy_coefficient` is requested; both assume rows of
        `u` sum to one.
    """

    def _init_u(self, X: NDArray) -> None:
        """Initialize `u`, `_centers` and `eta` from a FCM run."""
        fcm = FCM(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            m=self.m,
            error=self.error,
            random_state=self.random_state,
            distance=self.distance,
            distance_params=self.distance_params,
        )
        fcm.fit(X)
        self.rng = fcm.rng
        self.u = fcm.u
        self._centers = fcm.centers
        um = self.u**self.m
        d2 = FCM._dist(X, self._centers, self.distance, self.distance_params)
        self.eta = (um * d2**2).sum(axis=0) / um.sum(axis=0)

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def soft_predict(self, X: NDArray) -> NDArray:
        """Typicality of each sample to each cluster

        Args:
            X (NDArray): New data to predict.

        Returns:
            NDArray: Typicality array with n_samples rows and n_clusters
            columns. Rows do not sum to one.
        """
        d2 = (
            FCM._dist(X, self._centers, self.distance, self.distance_params)
            ** 2
        )
        return 1.0 / (1.0 + (d2 / self.eta) ** (1 / (self.m - 1)))

    @property
    def partition_coefficient(self) -> float:
        """Not defined for possibilistic partitions."""
        raise NotImplementedError("Rows of `u` do not sum to one in PCM.")

    @property
    def partition_entropy_coefficient(self) -> float:
        """Not defined for possibilistic partitions."""
        raise NotImplementedError("Rows of `u` do not sum to one in PCM.")
