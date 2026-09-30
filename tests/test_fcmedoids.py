"""Tests for FCMedoids Class"""

# flake8: noqa
import numpy as np
import pytest

from fcmeans import FCMedoids

# three well separated blobs with known labels
rng = np.random.default_rng(0)
CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
X = np.vstack([c + rng.normal(scale=0.5, size=(30, 2)) for c in CENTERS])
Y = np.repeat(np.arange(3), 30)


def _is_pure(labels):
    """Each true blob maps to a single predicted cluster"""
    return all(len(set(labels[Y == k])) == 1 for k in range(3)) and (
        len(set(labels)) == 3
    )


def test_recovers_clusters_and_medoids_are_samples():
    """Test if the clusters are found and the centers are training samples"""
    fcmd = FCMedoids(n_clusters=3, random_state=42)
    fcmd.fit(X)
    assert _is_pure(fcmd.predict(X))
    assert fcmd.medoid_indices.shape == (3,)
    assert len(set(fcmd.medoid_indices)) == 3
    assert np.array_equal(fcmd.centers, X[fcmd.medoid_indices])


def test_dont_fit():
    """Test if its returns Exceptions"""
    fcmd = FCMedoids()
    with pytest.raises(ReferenceError):
        fcmd.centers


def test_more_clusters_than_samples():
    """Test if n_clusters greater than n_samples is a clear error"""
    fcmd = FCMedoids(n_clusters=11)
    with pytest.raises(ValueError, match="n_clusters"):
        fcmd.fit(X[:10])


def test_u_is_a_fuzzy_partition_and_medoids_belong_to_themselves():
    """Test if u is finite and sums to one, with no NaN at zero distance"""
    fcmd = FCMedoids(n_clusters=3, random_state=42)
    fcmd.fit(X)
    assert np.isfinite(fcmd.u).all()
    assert np.allclose(fcmd.u.sum(axis=1), 1.0)
    assert np.allclose(fcmd.soft_predict(X), fcmd.u)
    for i, k in enumerate(fcmd.medoid_indices):
        assert fcmd.u[k, i] == 1.0


def test_medoids_minimize_the_weighted_cost():
    """Test the medoid update against loops over all candidates"""
    m = 2.0
    fcmd = FCMedoids(n_clusters=3, m=m, random_state=42)
    fcmd.fit(X)
    for i in range(3):
        cost = [
            sum(
                fcmd.u[j, i] ** m * np.sum((X[j] - X[k]) ** 2)
                for j in range(len(X))
            )
            for k in range(len(X))
        ]
        assert np.argmin(cost) == fcmd.medoid_indices[i]


def test_u_matches_fcm_formula():
    """Test u of the non-medoid samples against the double-sum formula"""
    m = 2.0
    fcmd = FCMedoids(n_clusters=3, m=m, random_state=42)
    fcmd.fit(X)
    d = np.linalg.norm(X[:, None, :] - fcmd.centers, axis=2)
    for j in np.setdiff1d(np.arange(len(X)), fcmd.medoid_indices):
        for i in range(3):
            expected = 1 / sum(
                (d[j, i] / d[j, l]) ** (2 / (m - 1)) for l in range(3)
            )
            assert np.isclose(fcmd.u[j, i], expected)


def test_custom_distance():
    """Test if a user distance (here Manhattan) drives the medoid search"""

    def manhattan(A, B, params):
        return np.abs(A[:, None, :] - B).sum(axis=-1)

    fcmd = FCMedoids(n_clusters=3, distance=manhattan, random_state=42)
    fcmd.fit(X)
    assert _is_pure(fcmd.predict(X))
    assert np.array_equal(fcmd.centers, X[fcmd.medoid_indices])


def test_pairwise_distances_are_not_kept():
    """Test if the O(n^2) matrix is freed, so saved models stay small"""
    fcmd = FCMedoids(n_clusters=3, random_state=42)
    fcmd.fit(X)
    assert not hasattr(fcmd, "_d2")


def test_random_state_reproducible():
    """Test if the same seed gives the same medoids"""
    a = FCMedoids(n_clusters=3, random_state=7)
    b = FCMedoids(n_clusters=3, random_state=7)
    a.fit(X)
    b.fit(X)
    assert np.array_equal(a.medoid_indices, b.medoid_indices)
