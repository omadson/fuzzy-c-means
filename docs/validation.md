# Validity indices

FCM needs the number of clusters $c$ up front. Validity indices score a
fitted partition so that you can compare different values of $c$. All of
them use the euclidean distance, work on the output of `FCM` (or a variant
whose rows of `u` sum to one) and need $c \geq 2$.

| Function | Better | Measures |
| --- | --- | --- |
| `xie_beni` | lower | compactness over separation of the centers |
| `fukuyama_sugeno` | lower | compactness minus spread of the centers around their mean |
| `davies_bouldin` | lower | membership weighted scatter over center distance |
| `fuzzy_silhouette` | higher | silhouette weighted by how clear each membership is |

The equations and references are in the [reference](reference.md).
`fuzzy_silhouette` builds the $n \times n$ distance matrix, so it needs
$O(n^2)$ memory; the other three are linear in $n$.

## Usage

```python
import numpy as np
from fcmeans import FCM, select_n_clusters, xie_beni

X = np.vstack([np.random.normal(loc=c, size=(50, 2)) for c in (0, 8, 16)])

fcm = FCM(n_clusters=3, random_state=42)
fcm.fit(X)
xie_beni(X, fcm.u, fcm.centers, fcm.m)

# fit for each c and recommend one
best, scores = select_n_clusters(
    X, range(2, 8), index="fuzzy_silhouette", random_state=42
)
```

Extra keyword arguments of `select_n_clusters` go to the model, and `model`
selects another class (`model=GK`, for instance).

## Notes

Indices disagree, and none is reliable alone. Fukuyama-Sugeno in particular
tends to tie neighboring values of $c$ on well separated data. Compare a few
and look at the scores, not only at the winner.
