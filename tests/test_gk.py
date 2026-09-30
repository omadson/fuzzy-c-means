"""Tests for GK Class"""

# flake8: noqa
import numpy as np
import pytest
from pydantic import ValidationError

from fcmeans import FCM, GK

# two elongated parallel clusters: spherical FCM splits them left/right
rng = np.random.default_rng(0)
X = np.vstack(
    [
        np.c_[rng.normal(0, 10, 100), rng.normal(0, 0.5, 100)],
        np.c_[rng.normal(0, 10, 100), rng.normal(4, 0.5, 100)],
    ]
)
Y = np.repeat([0, 1], 100)


def _accuracy(labels):
    acc = (labels == Y).mean()
    return max(acc, 1 - acc)


def test_recovers_elongated_clusters_where_fcm_fails():
    """Test the point of GK: ellipsoidal clusters"""
    gk = GK(n_clusters=2, random_state=42)
    gk.fit(X)
    fcm = FCM(n_clusters=2, random_state=42)
    fcm.fit(X)
    assert _accuracy(gk.predict(X)) > 0.99
    assert _accuracy(fcm.predict(X)) < 0.7


def test_dont_fit():
    """Test if its returns Exceptions"""
    gk = GK()
    with pytest.raises(ReferenceError):
        gk.centers


def test_distance_is_not_configurable():
    """Test if a custom distance is rejected instead of ignored"""
    GK(distance="euclidean")
    with pytest.raises(ValidationError):
        GK(distance="cosine")
    with pytest.raises(ValidationError):
        GK(distance=lambda A, B, params: np.zeros((len(A), len(B))))


def test_u_rows_sum_to_one_and_soft_predict_matches():
    """Test if u is a fuzzy partition reproduced by soft_predict"""
    gk = GK(n_clusters=2, random_state=42)
    gk.fit(X)
    assert np.allclose(gk.u.sum(axis=1), 1.0)
    assert np.allclose(gk.soft_predict(X), gk.u)


def test_norm_matrices_have_unit_determinant():
    """Test the fixed-volume constraint det(A_i) = 1"""
    gk = GK(n_clusters=2, random_state=42)
    gk.fit(X)
    assert gk.norm_matrices.shape == (2, 2, 2)
    assert np.allclose(np.linalg.det(gk.norm_matrices), 1.0)


def test_norm_matrices_match_paper_formulas():
    """Test A_i after one iteration against loops over the samples"""
    m, c, seed = 2.0, 2, 42
    gk = GK(n_clusters=c, m=m, max_iter=1, reg=0.0, random_state=seed)
    gk.fit(X)
    # rebuild the initial partition the model started from
    u = np.random.default_rng(seed).uniform(size=(len(X), c))
    u /= u.sum(axis=1, keepdims=True)
    for i in range(c):
        w = u[:, i] ** m
        v = (w[:, None] * X).sum(axis=0) / w.sum()
        F = sum(w[j] * np.outer(X[j] - v, X[j] - v) for j in range(len(X)))
        F /= w.sum()
        A = np.linalg.det(F) ** (1 / X.shape[1]) * np.linalg.inv(F)
        assert np.allclose(gk.centers[i], v)
        assert np.allclose(gk.norm_matrices[i], A)


def test_degenerate_feature_stays_finite():
    """Test if a constant feature (singular covariance) does not give NaN"""
    gk = GK(n_clusters=2, random_state=42)
    gk.fit(np.c_[X, np.zeros(len(X))])
    assert np.isfinite(gk.u).all()
    assert np.isfinite(gk.centers).all()


def test_random_state_reproducible():
    """Test if the same seed gives the same centers"""
    a = GK(n_clusters=2, random_state=7)
    b = GK(n_clusters=2, random_state=7)
    a.fit(X)
    b.fit(X)
    assert np.array_equal(a.centers, b.centers)
