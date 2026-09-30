"""Tests for KFCM Class"""

# flake8: noqa
import math

import numpy as np
import pytest
from pydantic import ValidationError

from fcmeans import FCM, KFCM

# three well separated blobs with known labels
rng = np.random.default_rng(0)
CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
X = np.vstack([c + rng.normal(scale=0.5, size=(30, 2)) for c in CENTERS])
Y = np.repeat(np.arange(3), 30)
# the same blobs plus a small group of far outliers
OUTLIERS = np.array([25.0, 25.0]) + rng.normal(scale=0.3, size=(6, 2))
XO = np.vstack([X, OUTLIERS])


def _center_error(centers):
    """Distance from the worst recovered blob center to the closest center"""
    return max(np.linalg.norm(centers - c, axis=1).min() for c in CENTERS)


def test_recovers_clusters():
    """Test if each true blob maps to a single predicted cluster"""
    kfcm = KFCM(n_clusters=3, random_state=42)
    kfcm.fit(X)
    labels = kfcm.predict(X)
    for k in range(3):
        assert len(set(labels[Y == k])) == 1
    assert len(set(labels)) == 3
    assert _center_error(kfcm.centers) < 0.5


def test_outliers_barely_move_the_centers():
    """Test the kernel weighting: FCM centers are pulled, KFCM's are not"""
    kfcm = KFCM(n_clusters=3, random_state=0)
    kfcm.fit(XO)
    fcm = FCM(n_clusters=3, random_state=0)
    fcm.fit(XO)
    assert _center_error(kfcm.centers) < 0.3
    assert _center_error(fcm.centers) > 0.7


def test_dont_fit():
    """Test if its returns Exceptions"""
    kfcm = KFCM()
    with pytest.raises(ReferenceError):
        kfcm.centers


def test_invalid_parameters():
    """Test if gamma must be positive and distance stays the default"""
    with pytest.raises(ValidationError):
        KFCM(gamma=0.0)
    with pytest.raises(ValidationError):
        KFCM(distance="cosine")
    KFCM(gamma=1.0, distance="euclidean")


def test_gamma_default_and_explicit():
    """Test the 1 / (n_features * var) default and an explicit gamma"""
    auto = KFCM(n_clusters=3, random_state=42)
    auto.fit(X)
    assert np.isclose(auto.gamma_, 1 / (X.shape[1] * X.var()))
    fixed = KFCM(n_clusters=3, gamma=0.5, random_state=42)
    fixed.fit(X)
    assert fixed.gamma_ == 0.5


def test_u_matches_kernel_distance_formula():
    """Test u against d^2 = 2 (1 - K), written with loops"""
    m = 2.0
    kfcm = KFCM(n_clusters=3, m=m, random_state=42)
    kfcm.fit(X)
    n, c = len(X), 3
    one_minus_k = np.zeros((n, c))
    for j in range(n):
        for i in range(c):
            sq = sum((X[j] - kfcm.centers[i]) ** 2)
            one_minus_k[j, i] = 1 - math.exp(-kfcm.gamma_ * sq)
    for j in range(n):
        for i in range(c):
            expected = 1 / sum(
                (one_minus_k[j, i] / one_minus_k[j, l]) ** (1 / (m - 1))
                for l in range(c)
            )
            assert np.isclose(kfcm.u[j, i], expected)


def test_centers_are_kernel_weighted_means():
    """Test that centers are a fixed point of the update, with loops"""
    m = 2.0
    kfcm = KFCM(n_clusters=3, m=m, random_state=42)
    kfcm.fit(X)
    for i in range(3):
        w = [
            kfcm.u[j, i] ** m
            * math.exp(-kfcm.gamma_ * sum((X[j] - kfcm.centers[i]) ** 2))
            for j in range(len(X))
        ]
        v = sum(w[j] * X[j] for j in range(len(X))) / sum(w)
        assert np.allclose(kfcm.centers[i], v, atol=1e-4)


def test_u_rows_sum_to_one_and_soft_predict_matches():
    """Test if u is a fuzzy partition reproduced by soft_predict"""
    kfcm = KFCM(n_clusters=3, random_state=42)
    kfcm.fit(X)
    assert np.allclose(kfcm.u.sum(axis=1), 1.0)
    assert np.allclose(kfcm.soft_predict(X), kfcm.u)


def test_huge_gamma_stays_finite():
    """Test if a center that no sample reaches does not give NaN"""
    kfcm = KFCM(n_clusters=3, gamma=1e6, random_state=42)
    kfcm.fit(X)
    assert np.isfinite(kfcm.u).all()
    assert np.isfinite(kfcm.centers).all()


def test_random_state_reproducible():
    """Test if the same seed gives the same centers"""
    a = KFCM(n_clusters=3, random_state=7)
    b = KFCM(n_clusters=3, random_state=7)
    a.fit(X)
    b.fit(X)
    assert np.array_equal(a.centers, b.centers)
