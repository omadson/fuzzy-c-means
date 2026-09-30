# Fuzzy-possibilistic C-means (FPCM)

FCM memberships sum to one over the clusters, which makes them sensitive to
noise; PCM typicalities are free of that constraint but tend to make the
clusters collapse onto each other. FPCM[^1] keeps both: each sample $j$ has a
fuzzy membership $u_{ij}$ and a typicality $t_{ij}$ to cluster $i$, and the
centers use both.

The two matrices are normalized in different directions:

- $u$: rows sum to one, $\sum_i u_{ij} = 1$ (over clusters, as in FCM);
- $t$: columns sum to one, $\sum_j t_{ij} = 1$ (over the samples of each
  cluster).

$$
u_{ij} = \left[\sum_{l=1}^{c} \left(\frac{d_{ij}}{d_{lj}}\right)^{2/(m-1)}\right]^{-1}
\qquad
t_{ij} = \left[\sum_{k=1}^{n} \left(\frac{d_{ij}}{d_{ik}}\right)^{2/(\eta-1)}\right]^{-1}
$$

$$
v_i = \frac{\sum_j \left(u_{ij}^m + t_{ij}^\eta\right) x_j}
{\sum_j \left(u_{ij}^m + t_{ij}^\eta\right)}
$$

`m` and `eta` (both greater than 1) control the fuzziness of $u$ and $t$.

## Usage

```python
import numpy as np
from fcmeans import FPCM

X = np.random.normal(size=(100, 2))
fpcm = FPCM(n_clusters=3, m=2.0, eta=2.0, random_state=42)
fpcm.fit(X)
fpcm.centers  # cluster centers
fpcm.u  # fuzzy memberships of the training data, rows sum to one
fpcm.t  # typicalities of the training data, columns sum to one
fpcm.predict(X)  # cluster of highest membership
```

`FPCM` takes the same parameters as [`FCM`](../reference.md), plus `eta`.

## Notes

- `soft_predict` and `predict` use the fuzzy memberships only. Typicality is
  normalized over the training samples, so it is not defined for new data.
- Since $t$ sums to one over all $n$ samples, its values shrink as $n$ grows
  and $t_{ij}^\eta$ becomes negligible next to $u_{ij}^m$. On large data sets
  FPCM then behaves much like FCM.
- `partition_coefficient` and `partition_entropy_coefficient` are inherited
  and computed on $u$.

[^1]: Pal, N. R., K. Pal, and J. C. Bezdek. "[A mixed c-means clustering model.](https://doi.org/10.1109/FUZZY.1997.616338)" _Proceedings of 6th International Fuzzy Systems Conference_ 1 (1997): 11-21.
