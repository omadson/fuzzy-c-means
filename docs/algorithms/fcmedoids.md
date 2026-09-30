# Fuzzy C-medoids (FCMedoids)

FCM prototypes are weighted means, which need not be samples of the data set.
Fuzzy c-medoids[^1] restricts each prototype $v_i$ to be one of the samples,
its *medoid*, and minimizes the FCM objective under that constraint:

$$
J = \sum_{i=1}^{c} \sum_{j=1}^{n} u_{ij}^m \, d^2(x_j, v_i),
\qquad v_i \in X
$$

Memberships are updated as in FCM. Each medoid is the sample with the lowest
membership-weighted cost for its cluster:

$$
u_{ij} = \left[\sum_{l=1}^{c}
\left(\frac{d_{ij}}{d_{lj}}\right)^{2/(m-1)}\right]^{-1}
\qquad
v_i = \operatorname*{arg\,min}_{x_k \in X}
\sum_{j=1}^{n} u_{ij}^m \, d^2(x_j, x_k)
$$

Both steps lower $J$, and the search over samples is finite, so the algorithm
stops when the medoids stop changing.

## Usage

```python
import numpy as np
from fcmeans import FCMedoids

rng = np.random.default_rng(0)
centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
X = np.vstack([c + rng.normal(scale=0.5, size=(30, 2)) for c in centers])

fcmd = FCMedoids(n_clusters=3, random_state=42)
fcmd.fit(X)
fcmd.medoid_indices  # rows of X chosen as medoids
fcmd.centers  # the medoids themselves: X[fcmd.medoid_indices]
fcmd.predict(X)  # cluster of highest membership
```

`FCMedoids` takes the same parameters as [`FCM`](../reference.md).

## Custom distances

Because the search only compares distances, any `distance` of `FCM` works,
including a callable with signature `(A, B, distance_params) -> NDArray` that
returns the distance from every row of `A` to every row of `B`:

```python
def manhattan(A, B, params):
    return np.abs(A[:, None, :] - B).sum(axis=-1)


fcmd = FCMedoids(n_clusters=3, distance=manhattan, random_state=42)
fcmd.fit(X)
```

## Notes

- **Dissimilarity.** The dissimilarity in the objective is the *squared*
  distance $d^2$, so the memberships are exactly those of FCM. The original
  paper allows any dissimilarity $r$ and uses $r^{-1/(m-1)}$ in the membership
  update; with $r = d^2$ the two coincide.
- **Initialization.** The first medoid is a random sample, and each next one
  is drawn with probability proportional to its squared distance to the
  closest medoid already chosen (k-means++ seeding). The result can still be
  a poor local optimum. Fit with several `random_state` values and keep the
  best. In 50 runs on three well-separated blobs, 49 recovered all three.
- **Memory and time.** The distances between all pairs of training samples
  are computed once and dropped at the end of `fit`, so `fit` needs $O(n^2)$
  memory and each iteration costs $O(c\,n^2)$. It is meant for data sets of a
  few thousand samples.
- A sample that coincides with a medoid has distance zero, and belongs only to
  that cluster (split evenly if two medoids coincide). This is always the case
  for the medoids themselves, so `FCM.soft_predict` handles it.
- `n_clusters` greater than the number of samples raises `ValueError`.

[^1]: Krishnapuram, R., A. Joshi, O. Nasraoui, and L. Yi. "[Low-complexity fuzzy relational clustering algorithms for Web mining.](https://doi.org/10.1109/91.940971)" _IEEE Transactions on Fuzzy Systems_ 9.4 (2001): 595-607.
