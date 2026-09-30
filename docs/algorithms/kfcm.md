# Kernel fuzzy C-means (KFCM)

FCM measures distance with the Euclidean norm. Kernel FCM[^1] measures it
after mapping the samples to the feature space $\phi$ of a Gaussian kernel

$$
K(x, v) = \exp\left(-\gamma \|x - v\|^2\right)
\qquad
d^2(x, v) = \|\phi(x) - \phi(v)\|^2 = 2\,\bigl(1 - K(x, v)\bigr)
$$

The prototypes stay in the input space, so they are ordinary points and
`centers` and `predict` work as in FCM. Memberships are updated as in FCM,
with this distance, and centers are kernel-weighted means:

$$
u_{ij} = \left[\sum_{l=1}^{c}
\left(\frac{1 - K(x_j, v_i)}{1 - K(x_j, v_l)}\right)^{1/(m-1)}\right]^{-1}
\qquad
v_i = \frac{\sum_j u_{ij}^m K(x_j, v_i)\, x_j}{\sum_j u_{ij}^m K(x_j, v_i)}
$$

The kernel $K(x_j, v_i)$ in the center update is close to zero for samples far
from the center, so outliers hardly move it.

## Usage

```python
import numpy as np
from fcmeans import FCM, KFCM

rng = np.random.default_rng(0)
centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
blobs = np.vstack([c + rng.normal(scale=0.5, size=(30, 2)) for c in centers])
outliers = np.array([25.0, 25.0]) + rng.normal(scale=0.3, size=(6, 2))
X = np.vstack([blobs, outliers])

kfcm = KFCM(n_clusters=3, random_state=0)
kfcm.fit(X)
kfcm.centers  # stay on the three blobs
kfcm.gamma_  # kernel coefficient used
kfcm.predict(X)  # cluster of highest membership

fcm = FCM(n_clusters=3, random_state=0)
fcm.fit(X)
fcm.centers  # pulled towards the outliers
```

`KFCM` takes the same parameters as [`FCM`](../reference.md), plus `gamma`.

## Notes

- **`gamma`.** It sets the reach of the kernel: samples farther than about
  $1/\sqrt{\gamma}$ from a center get almost no weight. If left as `None`, it
  is set to `1 / (n_features * X.var())` at `fit` and stored in `gamma_`. As
  $\gamma \to 0$ the model approaches FCM. A very large `gamma` makes the
  memberships close to uniform.
- **Initialization.** Centers and memberships start from a FCM run with the
  same parameters, because a random start leaves the centers out of reach of
  the kernel when `gamma` is large. The consequence is that if FCM places a
  center on the outliers, KFCM keeps it there. This costs one extra FCM fit.
- **Gaussian kernel only.** The closed-form center update above is specific to
  the Gaussian kernel, and `distance` cannot be changed; passing a `distance`
  other than the default raises a `ValidationError`.
- **Prototypes in the input space.** The model does not separate clusters
  that are not separable by their centers, such as concentric rings.
  Kernel methods with prototypes in the feature space do, but they have no
  centers in the input space. See Graves and Pedrycz[^2] for a comparison.
- A center that no sample reaches (all kernel weights zero) stays where it is
  instead of producing `nan`.

[^1]: Wu, Z.-D., W.-X. Xie, and J.-P. Yu. "[Fuzzy C-means clustering algorithm based on kernel method.](https://doi.org/10.1109/ICCIMA.2003.1238099)" _Proceedings Fifth International Conference on Computational Intelligence and Multimedia Applications. ICCIMA 2003_ (2003): 49-54.
[^2]: Graves, D., and W. Pedrycz. "[Kernel-based fuzzy clustering and fuzzy clustering: A comparative experimental study.](https://doi.org/10.1016/j.fss.2009.10.021)" _Fuzzy Sets and Systems_ 161.4 (2010): 522-543.
