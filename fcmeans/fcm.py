import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist

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

    r: 
    Container for the Mersenne Twister pseudo-random number generator.
    
    Methods
    -------
    fit(X=None)
        Prints the animals name and what sound it makes

    _predict(X=None)
        Prints the animals name and what sound it makes

    predict(X=None)
        Prints the animals name and what sound it makes

    References
    ----------
    .. [1] `Pattern Recognition with Fuzzy Objective Function Algorithms
        <https://doi.org/10.1007/978-1-4757-0450-1>`_
    .. [2] `FCM: The fuzzy c-means clustering algorithm
        <https://doi.org/10.1016/0098-3004(84)90020-7>`_

    """
    
    def __init__(self, n_clusters=10, max_iter=150, m=2, error=1e-5, random_state=42):
        self.u, self.centers = None, None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state

    def fit(self, X):
        """Compute fuzzy C-means clustering.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training instances to cluster.
        """
        n_samples = X.shape[0]

        r = np.random.RandomState(self.random_state)
        self.u = r.rand(n_samples,self.n_clusters)
        self.u = self.u / np.tile(self.u.sum(axis=1)[np.newaxis].T,self.n_clusters)

        for iteration in range(self.max_iter):
            u_old = self.u.copy()

            self.centers = self.next_centers(X)
            self.u = self._predict(X)

            # Stopping rule
            if norm(self.u - u_old) < self.error:
                break

        return self

    def next_centers(self, X):
        """Update cluster centers"""
        um = self.u ** self.m
        return (X.T @ um / np.sum(um, axis=0)).T

    def _predict(self, X):
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
        power = float(2 / (self.m - 1))
        temp = cdist(X, self.centers) ** power
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(temp.shape[-1], axis=1)
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

        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        u = self._predict(X)
        return np.argmax(u, axis=-1)
