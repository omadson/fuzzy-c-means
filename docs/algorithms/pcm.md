# Possibilistic C-means (PCM)

FCM forces the memberships of each sample to sum to one, so an outlier still
belongs to some cluster with high membership and drags its center. PCM[^1]
drops that constraint: $u_{ij}$ is the *typicality* of sample $j$ to cluster
$i$, computed independently for each cluster.

$$
u_{ij} = \frac{1}{1 + \left(d_{ij}^2 / \eta_i\right)^{1/(m-1)}}
\qquad
\eta_i = \frac{\sum_j u_{ij}^m d_{ij}^2}{\sum_j u_{ij}^m}
$$

Centers are updated as in FCM. The $\eta_i$ and the starting centers come
from an initial FCM run and stay fixed while PCM iterates, because PCM has
no term that keeps the clusters apart and can otherwise collapse onto the
same center.

## Usage

```python
import numpy as np
from fcmeans import PCM

X = np.random.normal(size=(100, 2))
pcm = PCM(n_clusters=3, random_state=42)
pcm.fit(X)
pcm.centers  # cluster centers
pcm.soft_predict(X)  # typicalities, rows do not sum to one
pcm.predict(X)  # cluster of highest typicality
```

`PCM` takes the same parameters as [`FCM`](../reference.md).
`partition_coefficient` and `partition_entropy_coefficient` raise
`NotImplementedError`, since they assume rows of `u` sum to one.

[^1]: Krishnapuram, R., and J. M. Keller. "[A possibilistic approach to clustering.](https://doi.org/10.1109/91.227387)" _IEEE Transactions on Fuzzy Systems_ 1.2 (1993): 98-110.
