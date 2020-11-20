# fuzzy-c-means

![GitHub](https://img.shields.io/github/license/omadson/fuzzy-c-means.svg)
[![PyPI](https://img.shields.io/pypi/v/fuzzy-c-means.svg)](http://pypi.org/project/fuzzy-c-means/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/w/omadson/fuzzy-c-means.svg)](https://github.com/omadson/fuzzy-c-means/pulse)
[![GitHub last commit](https://img.shields.io/github/last-commit/omadson/fuzzy-c-means.svg)](https://github.com/omadson/fuzzy-c-means/commit/master)
[![Downloads](https://pepy.tech/badge/fuzzy-c-means)](https://pepy.tech/project/fuzzy-c-means)
[![DOI](https://zenodo.org/badge/186457481.svg)](https://zenodo.org/badge/latestdoi/186457481)


`fuzzy-c-means` is a Python module implementing the [Fuzzy C-means][1] clustering algorithm.

## instalation
the `fuzzy-c-means` package is available in [PyPI](https://pypi.org/project/fuzzy-c-means/). to install, simply type the following command:
```
pip install fuzzy-c-means
```

## basic usage
simple example of use the `fuzzy-c-means` to cluster a dataset in tree groups:
```Python
%matplotlib inline

from fcmeans import FCM
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt


# create artifitial dataset
n_samples = 5000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here
centers = [(-5, -5), (0, 0), (5, 5)]

X,_ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.5,
                  centers=centers, shuffle=False, random_state=42)

# fit the fuzzy-c-means
fcm = FCM(n_clusters=3)
fcm.fit(X)

# outputs
fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis=1)


# plot result
f, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].scatter(X[:,0], X[:,1], alpha=.1)
axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="s", s=100, c='white')
plt.show()
```

## how to cite fuzzy-c-means package
if you use `fuzzy-c-means` package in your paper, please cite it in your publication.
```
@software{dias2019fuzzy,
  author       = {Madson Luiz Dantas Dias},
  title        = {fuzzy-c-means: An implementation of Fuzzy $C$-means clustering algorithm.},
  month        = may,
  year         = 2019,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3066222},
  url          = {https://git.io/fuzzy-c-means}
}
```

### citations
 - [Gene-Based Clustering Algorithms: Comparison Between Denclue, Fuzzy-C, and BIRCH](https://doi.org/10.1177/1177932220909851)


## contributing

this project is open for contributions. here are some of the ways for you to contribute:
 - bug reports/fix
 - features requests
 - use-case demonstrations

to make a contribution, just fork this repository, push the changes in your fork, open up an issue, and make a pull request!

## contributors
 - [Madson Dias](https://github.com/omadson)
 - [Dirk Nachbar](https://github.com/dirknbr)
 - [Alberth Florêncio](https://github.com/zealberth)

[1]: https://doi.org/10.1016/0098-3004(84)90020-7
[2]: http://scikit-learn.org/
