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
