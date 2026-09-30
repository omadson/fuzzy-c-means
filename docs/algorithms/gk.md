# Gustafson-Kessel (GK)

FCM measures distance with the Euclidean norm, which implicitly assumes
spherical clusters of the same size. Gustafson and Kessel[^1] give each
cluster $i$ its own norm matrix $A_i$, learned from the data, so the clusters
can be ellipsoids with different orientations.

$$
F_i = \frac{\sum_j u_{ij}^m (x_j - v_i)(x_j - v_i)^\top}{\sum_j u_{ij}^m}
\qquad
A_i = \det(F_i)^{1/p} F_i^{-1}
\qquad
d_{ij}^2 = (x_j - v_i)^\top A_i (x_j - v_i)
$$

$F_i$ is the fuzzy covariance matrix of cluster $i$ and $p$ is the number of
features. The factor $\det(F_i)^{1/p}$ fixes $\det(A_i) = 1$, so clusters may
change shape but not volume. Centers and memberships are updated as in FCM,
using this distance.

## Usage

```python
import numpy as np
from fcmeans import GK

rng = np.random.default_rng(0)
# two elongated clusters, parallel to the x axis
X = np.vstack(
    [
        np.c_[rng.normal(0, 10, 100), rng.normal(0, 0.5, 100)],
        np.c_[rng.normal(0, 10, 100), rng.normal(4, 0.5, 100)],
    ]
)

gk = GK(n_clusters=2, random_state=42)
gk.fit(X)
gk.centers  # cluster centers
gk.norm_matrices  # one A_i per cluster, shape (2, 2, 2)
gk.predict(X)  # cluster of highest membership
```

On this data `FCM` splits the points left and right, across both clusters,
while `GK` recovers the two strips.

`GK` takes the same parameters as [`FCM`](../reference.md), plus `reg`.

## Notes

- `distance` and `distance_params` cannot be changed: the adaptive norm
  replaces them. Passing a `distance` other than the default raises a
  `ValidationError`.
- A covariance matrix is singular when a cluster has a constant or collinear
  feature, or fewer samples than features. `reg` (default `1e-6`) adds
  `reg * trace(F_i) / p` to the diagonal of each $F_i$ so it stays
  invertible. Set `reg=0` for the algorithm exactly as published.
- The cluster volume is fixed at one ($\rho_i = 1$ in the original paper);
  it is not configurable.
- Each iteration estimates $c$ covariance matrices of size $p \times p$, so
  GK needs many more samples per cluster than FCM and is slower as the
  number of features grows.

[^1]: Gustafson, D. E., and W. C. Kessel. "[Fuzzy clustering with a fuzzy covariance matrix.](https://doi.org/10.1109/CDC.1978.268028)" _1978 IEEE Conference on Decision and Control including the 17th Symposium on Adaptive Processes_ (1978): 761-766.
