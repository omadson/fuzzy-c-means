"""Tests for FCM Class"""

# flake8: noqa
import numpy as np
import pytest

from fcmeans import FCM

# some test data
X = np.random.normal(size=(10, 2))


def test_u_creation():
    """Test if its generate u matrix"""
    fcm = FCM()
    fcm.fit(X)
    assert fcm.u is not None


def test_dont_fit():
    """Test if its returns Exceptions"""
    fcm = FCM()
    assert fcm.trained == False
    with pytest.raises(ReferenceError):
        partition_entropy_coefficient = fcm.partition_entropy_coefficient
    with pytest.raises(ReferenceError):
        partition_coefficient = fcm.partition_coefficient
    with pytest.raises(ReferenceError):
        centers = fcm.centers


def test_u_rows_sum_to_one():
    """Test if membership matrix rows are normalized"""
    fcm = FCM(n_clusters=3, random_state=42)
    fcm.fit(X)
    assert np.allclose(fcm.u.sum(axis=1), 1.0)


def test_soft_predict_matches_u_after_fit():
    """Test if soft_predict(X) reproduces the fitted membership matrix"""
    fcm = FCM(n_clusters=3, random_state=42)
    fcm.fit(X)
    assert np.allclose(fcm.soft_predict(X), fcm.u)


def test_predict_shape_and_range():
    """Test if predict returns one deterministic label per sample"""
    n_clusters = 3
    fcm = FCM(n_clusters=n_clusters, random_state=42)
    fcm.fit(X)
    labels = fcm.predict(X)
    assert labels.shape == (X.shape[0],)
    assert labels.min() >= 0
    assert labels.max() < n_clusters
    assert np.array_equal(labels, fcm.predict(X))


def test_sample_on_a_center_belongs_only_to_it():
    """Test if zero distance gives a crisp membership instead of NaN"""
    fcm = FCM(n_clusters=3, random_state=42)
    fcm.fit(X)
    assert np.allclose(fcm.soft_predict(fcm.centers), np.eye(3))


def test_minkowski_distance():
    """Test if minkowski distance uses absolute differences"""
    A = np.array([[1.0, -1.0], [3.0, 4.0]])
    B = np.array([[0.0, 0.0]])
    assert np.allclose(FCM._minkowski(A, B, 1.0), [[2.0], [7.0]])
    assert np.allclose(
        FCM._minkowski(A, B, 3.0), [[2 ** (1 / 3)], [91 ** (1 / 3)]]
    )


def test_init_rejects_unknown_value():
    """Test if `init` only accepts the implemented strategies"""
    with pytest.raises(ValueError):
        FCM(init="kmeans")


def test_kmeans_plusplus_seeds_every_blob():
    """Test if k-means++ puts one seed in each well-separated blob"""
    rng = np.random.default_rng(0)
    blobs = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    data = np.vstack([c + rng.normal(size=(30, 2)) for c in blobs])
    for seed in range(10):
        fcm = FCM(n_clusters=3, init="k-means++", random_state=seed)
        fcm.fit(data)
        assert np.array_equal(np.sort(fcm.predict(blobs)), np.arange(3)), seed


def test_kmeans_plusplus_u_and_reproducibility():
    """Test if the seeded partition is valid and set by random_state"""
    a = FCM(n_clusters=3, init="k-means++", random_state=7)
    b = FCM(n_clusters=3, init="k-means++", random_state=7)
    a._init_u(X)
    b._init_u(X)
    assert np.allclose(a.u.sum(axis=1), 1.0)
    assert np.array_equal(a.u, b.u)
    assert (a._centers[:, None] == X).all(axis=2).any(axis=1).all()


def test_kmeans_plusplus_duplicated_points():
    """Test if identical samples (zero total cost) do not break seeding"""
    fcm = FCM(n_clusters=3, init="k-means++", random_state=0)
    fcm.fit(np.ones((10, 2)))
    assert not np.isnan(fcm.u).any()
