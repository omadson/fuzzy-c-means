"""Tests for PCM Class"""

# flake8: noqa
import numpy as np
import pytest

from fcmeans import FCM, PCM

# three well separated blobs with known labels
rng = np.random.default_rng(0)
CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
X = np.vstack([c + rng.normal(scale=0.5, size=(30, 2)) for c in CENTERS])
Y = np.repeat(np.arange(3), 30)


def test_recovers_clusters():
    """Test if each true blob maps to a single predicted cluster"""
    pcm = PCM(n_clusters=3, random_state=42)
    pcm.fit(X)
    labels = pcm.predict(X)
    for k in range(3):
        assert len(set(labels[Y == k])) == 1
    assert len(set(labels)) == 3
    for c in CENTERS:
        assert np.linalg.norm(pcm.centers - c, axis=1).min() < 0.5


def test_dont_fit():
    """Test if its returns Exceptions"""
    pcm = PCM()
    with pytest.raises(ReferenceError):
        pcm.centers


def test_validity_indices_not_implemented():
    """Test if indices that assume sum(u) == 1 are rejected"""
    pcm = PCM(n_clusters=3, random_state=42)
    pcm.fit(X)
    with pytest.raises(NotImplementedError):
        pcm.partition_coefficient
    with pytest.raises(NotImplementedError):
        pcm.partition_entropy_coefficient


def test_u_in_unit_interval_and_rows_not_normalized():
    """Test if u is a typicality matrix, not a partition"""
    pcm = PCM(n_clusters=3, random_state=42)
    pcm.fit(X)
    assert pcm.u.shape == (X.shape[0], 3)
    assert ((pcm.u > 0) & (pcm.u <= 1)).all()
    assert not np.allclose(pcm.u.sum(axis=1), 1.0)


def test_outlier_has_low_typicality_but_fcm_does_not():
    """Test the defining property: outliers belong to no cluster"""
    Xo = np.vstack([X, [[100.0, 100.0]]])
    pcm = PCM(n_clusters=3, random_state=42)
    pcm.fit(Xo)
    fcm = FCM(n_clusters=3, random_state=42)
    fcm.fit(Xo)
    assert pcm.soft_predict(Xo)[-1].max() < 0.01
    assert fcm.soft_predict(Xo)[-1].max() > 0.3


def test_soft_predict_matches_u_after_fit():
    """Test if soft_predict(X) reproduces the fitted typicality matrix"""
    pcm = PCM(n_clusters=3, random_state=42)
    pcm.fit(X)
    assert np.allclose(pcm.soft_predict(X), pcm.u)


def test_random_state_reproducible():
    """Test if the same seed gives the same centers"""
    a = PCM(n_clusters=3, random_state=7)
    b = PCM(n_clusters=3, random_state=7)
    a.fit(X)
    b.fit(X)
    assert np.array_equal(a.centers, b.centers)
