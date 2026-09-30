"""Tests for the validity indices"""

# flake8: noqa
import numpy as np
import pytest

from fcmeans import FCM
from fcmeans import validation as v

# three well separated blobs with known labels
rng = np.random.default_rng(0)
CENTERS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
X = np.vstack([c + rng.normal(scale=0.5, size=(30, 2)) for c in CENTERS])
M = 2.0

fcm = FCM(n_clusters=3, random_state=42)
fcm.fit(X)
U, V = fcm.u, fcm.centers


def test_xie_beni_matches_loops():
    """Test if XB equals the plain-loop transcription of the paper"""
    num = sum(
        U[k, i] ** M * np.sum((X[k] - V[i]) ** 2)
        for i in range(3)
        for k in range(len(X))
    )
    sep = min(
        np.sum((V[i] - V[j]) ** 2)
        for i in range(3)
        for j in range(3)
        if i != j
    )
    assert np.isclose(v.xie_beni(X, U, V, M), num / (len(X) * sep))


def test_fukuyama_sugeno_matches_loops():
    """Test if FS equals the plain-loop transcription of the paper"""
    vbar = V.mean(axis=0)
    fs = sum(
        U[k, i] ** M
        * (np.sum((X[k] - V[i]) ** 2) - np.sum((V[i] - vbar) ** 2))
        for i in range(3)
        for k in range(len(X))
    )
    assert np.isclose(v.fukuyama_sugeno(X, U, V, M), fs)


def test_davies_bouldin_matches_loops():
    """Test if DB equals the plain-loop transcription"""
    S = [
        np.sqrt(
            sum(
                U[k, i] ** M * np.sum((X[k] - V[i]) ** 2)
                for k in range(len(X))
            )
            / sum(U[k, i] ** M for k in range(len(X)))
        )
        for i in range(3)
    ]
    db = np.mean(
        [
            max(
                (S[i] + S[j]) / np.linalg.norm(V[i] - V[j])
                for j in range(3)
                if j != i
            )
            for i in range(3)
        ]
    )
    assert np.isclose(v.davies_bouldin(X, U, V, M), db)


def test_fuzzy_silhouette_matches_loops():
    """Test if the fuzzy silhouette equals the plain-loop transcription"""
    labels = U.argmax(axis=1)
    n = len(X)
    s = np.zeros(n)
    for k in range(n):
        dist = lambda mask: np.mean(np.linalg.norm(X[mask] - X[k], axis=1))
        same = (labels == labels[k]) & (np.arange(n) != k)
        a = dist(same)
        b = min(dist(labels == j) for j in range(3) if j != labels[k])
        s[k] = (b - a) / max(a, b)
    top2 = np.sort(U, axis=1)[:, -2:]
    w = top2[:, 1] - top2[:, 0]
    assert np.isclose(v.fuzzy_silhouette(X, U, V), np.sum(w * s) / np.sum(w))


def test_fuzzy_silhouette_crisp_partition_is_plain_silhouette():
    """Test if a one-hot u gives weights of one and a score in [-1, 1]"""
    crisp = np.eye(3)[np.repeat(np.arange(3), 30)]
    score = v.fuzzy_silhouette(X, crisp, CENTERS)
    assert 0.9 < score <= 1.0


def test_index_direction_on_good_and_bad_partitions():
    """Test if the right c beats a wrong one on every index"""
    bad = FCM(n_clusters=2, random_state=42)
    bad.fit(X)
    for f in (v.xie_beni, v.fukuyama_sugeno, v.davies_bouldin):
        assert f(X, U, V, M) < f(X, bad.u, bad.centers, M)
    assert v.fuzzy_silhouette(X, U, V) > v.fuzzy_silhouette(
        X, bad.u, bad.centers
    )


@pytest.mark.parametrize(
    "index",
    ["xie_beni", "davies_bouldin", "fuzzy_silhouette"],
)
def test_select_n_clusters_finds_three_blobs(index):
    """Test if the scan recommends the true number of clusters"""
    best, scores = v.select_n_clusters(
        X, range(2, 7), index=index, random_state=42
    )
    assert best == 3
    assert sorted(scores) == [2, 3, 4, 5, 6]


def test_errors():
    """Test if invalid inputs are rejected"""
    with pytest.raises(ValueError):
        v.select_n_clusters(X, index="nope")
    with pytest.raises(ValueError):
        v.xie_beni(X, U[:, :1], V[:1])
    with pytest.raises(ValueError):
        v.fuzzy_silhouette(X, np.eye(3)[np.zeros(len(X), dtype=int)], V)


def test_select_n_clusters_fukuyama_sugeno_prefers_three_to_two():
    """Test the scan with FS, which nearly ties 3 and 4 on these blobs"""
    _, scores = v.select_n_clusters(
        X, range(2, 5), index="fukuyama_sugeno", random_state=42
    )
    assert scores[3] < scores[2]
