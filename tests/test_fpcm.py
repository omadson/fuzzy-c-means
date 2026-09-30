"""Tests for FPCM Class"""

# flake8: noqa
import numpy as np
import pytest
from pydantic import ValidationError

from fcmeans import FPCM

# three well separated blobs with known labels
rng = np.random.default_rng(0)
CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
X = np.vstack([c + rng.normal(scale=0.5, size=(30, 2)) for c in CENTERS])
Y = np.repeat(np.arange(3), 30)


def test_recovers_clusters():
    """Test if each true blob maps to a single predicted cluster"""
    fpcm = FPCM(n_clusters=3, random_state=42)
    fpcm.fit(X)
    labels = fpcm.predict(X)
    for k in range(3):
        assert len(set(labels[Y == k])) == 1
    assert len(set(labels)) == 3
    for c in CENTERS:
        assert np.linalg.norm(fpcm.centers - c, axis=1).min() < 0.5


def test_dont_fit():
    """Test if its returns Exceptions"""
    fpcm = FPCM()
    with pytest.raises(ReferenceError):
        fpcm.centers


def test_invalid_eta():
    """Test if eta must be greater than one"""
    with pytest.raises(ValidationError):
        FPCM(eta=1.0)


def test_u_rows_and_t_columns_sum_to_one():
    """Test the two normalizations: u over clusters, t over samples"""
    fpcm = FPCM(n_clusters=3, random_state=42)
    fpcm.fit(X)
    assert fpcm.t.shape == fpcm.u.shape == (X.shape[0], 3)
    assert np.allclose(fpcm.u.sum(axis=1), 1.0)
    assert np.allclose(fpcm.t.sum(axis=0), 1.0)
    assert ((fpcm.t > 0) & (fpcm.t < 1)).all()


def test_u_and_t_match_paper_formulas():
    """Test u and t against the double-sum form of the update equations"""
    fpcm = FPCM(n_clusters=3, m=2.0, eta=3.0, random_state=42)
    fpcm.fit(X)
    d = np.linalg.norm(X[:, None, :] - fpcm.centers, axis=2)
    n, c = d.shape
    u = np.zeros((n, c))
    t = np.zeros((n, c))
    for i in range(c):
        for k in range(n):
            u[k, i] = 1 / sum((d[k, i] / d[k, j]) ** 2 for j in range(c))
            t[k, i] = 1 / sum((d[k, i] / d[l, i]) ** 1 for l in range(n))
    assert np.allclose(fpcm.u, u)
    assert np.allclose(fpcm.t, t)


def test_soft_predict_matches_u_after_fit():
    """Test if soft_predict(X) reproduces the fitted membership matrix"""
    fpcm = FPCM(n_clusters=3, random_state=42)
    fpcm.fit(X)
    assert np.allclose(fpcm.soft_predict(X), fpcm.u)


def test_partition_coefficient_range():
    """Test if the inherited index stays in [1/c, 1]"""
    fpcm = FPCM(n_clusters=3, random_state=42)
    fpcm.fit(X)
    assert 1 / 3 <= fpcm.partition_coefficient <= 1.0


def test_random_state_reproducible():
    """Test if the same seed gives the same centers"""
    a = FPCM(n_clusters=3, random_state=7)
    b = FPCM(n_clusters=3, random_state=7)
    a.fit(X)
    b.fit(X)
    assert np.array_equal(a.centers, b.centers)
