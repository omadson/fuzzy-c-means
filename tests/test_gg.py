"""Tests for GG Class"""

# flake8: noqa
import numpy as np
import pytest
from pydantic import ValidationError

from fcmeans import FCM, GG, GK

# a large diffuse cluster next to a small dense one: FCM and GK split the
# diffuse one in two, GG keeps clusters of different size and density apart
rng = np.random.default_rng(0)
X = np.vstack(
    [
        rng.normal([0, 0], 3.0, size=(300, 2)),
        rng.normal([5, 0], 0.4, size=(40, 2)),
    ]
)
Y = np.repeat([0, 1], [300, 40])


def _accuracy(labels):
    acc = (labels == Y).mean()
    return max(acc, 1 - acc)


def test_recovers_clusters_of_different_size_and_density():
    """Test the point of GG: what FCM and GK cannot separate"""
    gg = GG(n_clusters=2, random_state=42)
    gg.fit(X)
    assert _accuracy(gg.predict(X)) > 0.9
    for other in (FCM, GK):
        model = other(n_clusters=2, random_state=42)
        model.fit(X)
        assert _accuracy(model.predict(X)) < 0.8


def test_dont_fit():
    """Test if its returns Exceptions"""
    gg = GG()
    with pytest.raises(ReferenceError):
        gg.centers


def test_distance_is_not_configurable():
    """Test if a custom distance is rejected instead of ignored"""
    with pytest.raises(ValidationError):
        GG(distance="cosine")


def test_u_rows_sum_to_one_and_soft_predict_matches():
    """Test if u is a fuzzy partition reproduced by soft_predict"""
    gg = GG(n_clusters=2, random_state=42)
    gg.fit(X)
    assert np.allclose(gg.u.sum(axis=1), 1.0)
    assert np.allclose(gg.soft_predict(X), gg.u)


def test_priors_sum_to_one_and_follow_cluster_size():
    """Test if the larger cluster gets the larger prior"""
    gg = GG(n_clusters=2, random_state=42)
    gg.fit(X)
    assert gg.priors.shape == (2,)
    assert np.isclose(gg.priors.sum(), 1.0)
    assert gg.priors.max() > 0.7


def test_u_matches_paper_formulas():
    """Test u against the distance of the paper, written with loops

    The direct formula overflows for samples far from a cluster, so only the
    rows where it stays finite are compared.
    """
    m = 2.0
    gg = GG(n_clusters=2, m=m, random_state=42)
    gg.fit(X)
    n, c = len(X), 2
    d2 = np.zeros((n, c))
    with np.errstate(over="ignore"):
        for i in range(c):
            F = gg.covariances[i]
            for j in range(n):
                diff = X[j] - gg.centers[i]
                d2[j, i] = (
                    np.sqrt(np.linalg.det(F))
                    / gg.priors[i]
                    * np.exp(0.5 * diff @ np.linalg.inv(F) @ diff)
                )
    ok = np.isfinite(d2).all(axis=1)
    assert ok.sum() > n // 2
    u = np.zeros((ok.sum(), c))
    for i in range(c):
        for j, row in enumerate(np.flatnonzero(ok)):
            u[j, i] = 1 / sum(
                (d2[row, i] / d2[row, l]) ** (1 / (m - 1)) for l in range(c)
            )
    assert np.allclose(gg.u[ok], u)


def test_covariances_and_priors_match_paper_formulas():
    """Test F_i and alpha_i after one iteration against loops"""
    m, c, seed = 2.0, 2, 42
    gg = GG(n_clusters=c, m=m, max_iter=1, reg=0.0, random_state=seed)
    gg.fit(X)
    # rebuild the GK partition the model started from
    gk = GK(n_clusters=c, m=m, max_iter=1, reg=0.0, random_state=seed)
    gk.fit(X)
    u = gk.u
    for i in range(c):
        w = u[:, i] ** m
        v = (w[:, None] * X).sum(axis=0) / w.sum()
        F = sum(w[j] * np.outer(X[j] - v, X[j] - v) for j in range(len(X)))
        F /= w.sum()
        assert np.allclose(gg.centers[i], v)
        assert np.allclose(gg.covariances[i], F)
        assert np.isclose(gg.priors[i], w.sum() / (u**m).sum())


def test_far_away_point_does_not_overflow():
    """Test if exp() of a huge Mahalanobis distance stays finite"""
    gg = GG(n_clusters=2, random_state=42)
    gg.fit(X)
    u = gg.soft_predict(np.array([[1e3, 1e3], [-1e3, 5e2]]))
    assert np.isfinite(u).all()
    assert np.allclose(u.sum(axis=1), 1.0)


def test_random_state_reproducible():
    """Test if the same seed gives the same centers"""
    a = GG(n_clusters=2, random_state=7)
    b = GG(n_clusters=2, random_state=7)
    a.fit(X)
    b.fit(X)
    assert np.array_equal(a.centers, b.centers)
