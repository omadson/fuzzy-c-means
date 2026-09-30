# Gath-Geva (GG)

[GK](gk.md) lets each cluster have its own shape, but all clusters keep the
same volume and the distance ignores how many points each one holds. Gath and
Geva[^1] add a prior probability $\alpha_i$ per cluster and use the inverse of
a Gaussian density as the distance, so clusters can differ in shape, size and
density.

$$
\alpha_i = \frac{\sum_j u_{ij}^m}{\sum_{l,j} u_{lj}^m}
\qquad
F_i = \frac{\sum_j u_{ij}^m (x_j - v_i)(x_j - v_i)^\top}{\sum_j u_{ij}^m}
$$

$$
d_{ij}^2 = \frac{\sqrt{\det F_i}}{\alpha_i}
\exp\left(\tfrac{1}{2} (x_j - v_i)^\top F_i^{-1} (x_j - v_i)\right)
$$

$F_i$ is the fuzzy covariance matrix of cluster $i$. Centers are updated as in
FCM, and memberships are computed from $d_{ij}^2$ with the FCM formula:

$$
u_{ij} = \left[\sum_{l=1}^{c}
\left(\frac{d_{ij}^2}{d_{lj}^2}\right)^{1/(m-1)}\right]^{-1}
$$

## Usage

```python
import numpy as np
from fcmeans import GG

rng = np.random.default_rng(0)
# a large diffuse cluster next to a small dense one
X = np.vstack(
    [
        rng.normal([0, 0], 3.0, size=(300, 2)),
        rng.normal([5, 0], 0.4, size=(40, 2)),
    ]
)

gg = GG(n_clusters=2, random_state=42)
gg.fit(X)
gg.centers  # cluster centers
gg.covariances  # one F_i per cluster, shape (2, 2, 2)
gg.priors  # alpha_i, sums to one
gg.predict(X)  # cluster of highest membership
```

On this data `FCM` and `GK` split the diffuse cluster in two, while `GG` keeps
the two clusters apart.

`GG` takes the same parameters as [`GK`](gk.md): those of
[`FCM`](../reference.md) plus `reg`.

## Notes

- **Initialization.** The exponential distance makes GG very sensitive to its
  starting point, and a random partition can end in a poor solution or a
  degenerate cluster. `fit` therefore runs a [GK](gk.md) model first, with the
  same parameters, and starts the GG iterations from its partition. This
  costs one extra GK fit.
- **Numerical range.** $d_{ij}^2$ grows as the exponential of the squared
  Mahalanobis distance and overflows for samples far from a cluster. The
  memberships are computed in the log domain, so `soft_predict` stays finite
  for any input.
- The constant $(2\pi)^{p/2}$ that appears in some presentations of the
  distance is the same for every cluster and cancels in the memberships, so
  it is left out.
- $F_i$ is regularized as in GK (`reg`, default `1e-6`). A cluster that
  collapses onto a single point still has a zero covariance matrix, and `fit`
  then fails with `numpy.linalg.LinAlgError`.
- `distance` and `distance_params` cannot be changed, as in GK.
- Each $F_i$ has $p(p+1)/2$ free parameters, so GG needs many samples per
  cluster, more so as the number of features grows.
- `partition_coefficient` and `partition_entropy_coefficient` are inherited
  and computed on $u$.

[^1]: Gath, I., and A. B. Geva. "[Unsupervised optimal fuzzy clustering.](https://doi.org/10.1109/34.192473)" _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 11.7 (1989): 773-780.
