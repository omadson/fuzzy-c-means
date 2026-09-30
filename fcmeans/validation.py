"""Cluster validity indices for fuzzy partitions.

All indices take the data `X` (n x d), the membership matrix `u` (n x c) and
the cluster centers (c x d) of a trained model, and use the euclidean
distance. They need at least two clusters.
"""

from collections.abc import Iterable
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .main import FCM


def _sq_dist(A: NDArray, B: NDArray) -> NDArray:
    """Squared euclidean distance between the rows of A and the rows of B."""
    return FCM._euclidean(A, B) ** 2


def _sq_center_dist(centers: NDArray) -> NDArray:
    """Squared distance between centers, with `inf` on the diagonal."""
    if len(centers) < 2:
        raise ValueError("Validity indices need at least two clusters.")
    d2 = _sq_dist(centers, centers)
    np.fill_diagonal(d2, np.inf)
    return d2


def xie_beni(
    X: NDArray, u: NDArray, centers: NDArray, m: float = 2.0
) -> float:
    r"""Xie-Beni index (lower is better).

    $$
    XB = \frac{\sum_{i,k} u_{ik}^m \lVert x_k - v_i \rVert^2}
    {n \min_{i \neq j} \lVert v_i - v_j \rVert^2}
    $$

    Xie, X. L., and G. Beni. "[A validity measure for fuzzy
    clustering.](https://doi.org/10.1109/34.85677)" _IEEE TPAMI_ 13.8
    (1991): 841-847.
    """
    d2c = _sq_center_dist(centers)
    compactness = np.sum(u**m * _sq_dist(X, centers))
    return float(compactness / (len(X) * d2c.min()))


def fukuyama_sugeno(
    X: NDArray, u: NDArray, centers: NDArray, m: float = 2.0
) -> float:
    r"""Fukuyama-Sugeno index (lower is better).

    $$
    FS = \sum_{i,k} u_{ik}^m \left(\lVert x_k - v_i \rVert^2 -
    \lVert v_i - \bar{v} \rVert^2\right)
    $$

    where $\bar{v}$ is the mean of the centers.

    Fukuyama, Y., and M. Sugeno. "A new method of choosing the number of
    clusters for the fuzzy c-means method." _Proc. 5th Fuzzy Systems
    Symposium_ (1989): 247-250.
    """
    _sq_center_dist(centers)
    spread = np.sum((centers - centers.mean(axis=0)) ** 2, axis=1)
    return float(np.sum(u**m * (_sq_dist(X, centers) - spread)))


def davies_bouldin(
    X: NDArray, u: NDArray, centers: NDArray, m: float = 2.0
) -> float:
    r"""Fuzzy Davies-Bouldin index (lower is better).

    The crisp scatter of each cluster is replaced by a membership weighted
    one, $S_i = \sqrt{\sum_k u_{ik}^m \lVert x_k - v_i \rVert^2 /
    \sum_k u_{ik}^m}$, and

    $$
    DB = \frac{1}{c} \sum_i \max_{j \neq i}
    \frac{S_i + S_j}{\lVert v_i - v_j \rVert}
    $$

    Davies, D. L., and D. W. Bouldin. "[A cluster separation
    measure.](https://doi.org/10.1109/TPAMI.1979.4766909)" _IEEE TPAMI_ 1.2
    (1979): 224-227.
    """
    d2c = _sq_center_dist(centers)
    um = u**m
    scatter = np.sqrt(np.sum(um * _sq_dist(X, centers), axis=0) / um.sum(0))
    with np.errstate(divide="ignore"):
        ratio = (scatter[:, None] + scatter) / np.sqrt(d2c)
    np.fill_diagonal(ratio, -np.inf)
    return float(ratio.max(axis=1).mean())


def fuzzy_silhouette(
    X: NDArray,
    u: NDArray,
    centers: NDArray,
    m: float = 2.0,
    alpha: float = 1.0,
) -> float:
    r"""Fuzzy silhouette (higher is better).

    The silhouette $s_k$ of each sample is computed on the crisp partition
    given by `u.argmax(axis=1)` and averaged with weights
    $(u_{pk} - u_{qk})^\alpha$, where $u_{pk}$ and $u_{qk}$ are the two
    largest memberships of sample $k$. Samples with no clear cluster weigh
    less. `m` is unused and only kept for a common signature.

    Builds the n x n distance matrix, so it needs O(n^2) memory.

    Campello, R. J. G. B., and E. R. Hruschka. "[A fuzzy extension of the
    silhouette width criterion for cluster
    analysis.](https://doi.org/10.1016/j.fss.2006.07.006)" _Fuzzy Sets and
    Systems_ 157.21 (2006): 2858-2875.
    """
    _sq_center_dist(centers)
    n, c = u.shape
    labels = u.argmax(axis=1)
    onehot = np.eye(c)[labels]
    counts = onehot.sum(axis=0)
    if (counts > 0).sum() < 2:
        raise ValueError(
            "The partition needs at least two non-empty clusters."
        )

    sq = np.sum(X**2, axis=1)
    D = np.sqrt(np.maximum(sq[:, None] + sq - 2 * X @ X.T, 0.0))
    sums = D @ onehot  # total distance from each sample to each cluster
    own = np.arange(n), labels
    with np.errstate(divide="ignore", invalid="ignore"):
        a = sums[own] / (counts[labels] - 1)
        mean_to = sums / counts
    mean_to[own] = np.inf
    b = mean_to.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (b - a) / np.maximum(a, b)
    s[(counts[labels] == 1) | ~np.isfinite(s)] = 0.0  # singleton or all equal

    top2 = np.sort(u, axis=1)[:, -2:]
    w = (top2[:, 1] - top2[:, 0]) ** alpha
    return float(np.sum(w * s) / np.sum(w))


# name -> (function, True when higher is better)
INDICES: dict[str, tuple[Callable[..., float], bool]] = {
    "xie_beni": (xie_beni, False),
    "fukuyama_sugeno": (fukuyama_sugeno, False),
    "davies_bouldin": (davies_bouldin, False),
    "fuzzy_silhouette": (fuzzy_silhouette, True),
}


def select_n_clusters(
    X: NDArray,
    n_clusters: Iterable[int] = range(2, 11),
    index: str = "xie_beni",
    model: type[FCM] = FCM,
    **params,
) -> tuple[int, dict[int, float]]:
    """Fit `model` for each number of clusters and score it with `index`.

    Args:
        X (NDArray): Data to cluster.
        n_clusters (Iterable[int]): Numbers of clusters to try, each >= 2.
        index (str): Name of a function in `INDICES`, such as `xie_beni`.
        model (type[FCM]): `FCM` or a variant whose `u` rows sum to one.
        **params: Passed to `model`, for example `m` or `random_state`.

    Returns:
        tuple[int, dict[int, float]]: The best number of clusters and the
        score of each one.
    """
    if index not in INDICES:
        raise ValueError(
            f"Unknown index {index!r}, use one of {list(INDICES)}"
        )
    func, higher_is_better = INDICES[index]
    scores = {}
    for c in n_clusters:
        fcm = model(n_clusters=c, **params)
        fcm.fit(X)
        scores[c] = func(X, fcm.u, fcm.centers, fcm.m)
    pick = max if higher_is_better else min
    return pick(scores, key=scores.__getitem__), scores
