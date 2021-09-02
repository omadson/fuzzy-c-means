# fuzzy-c-means

![GitHub](https://img.shields.io/github/license/omadson/fuzzy-c-means.svg)
[![PyPI](https://img.shields.io/pypi/v/fuzzy-c-means.svg)](http://pypi.org/project/fuzzy-c-means/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/w/omadson/fuzzy-c-means.svg)](https://github.com/omadson/fuzzy-c-means/pulse)
[![GitHub last commit](https://img.shields.io/github/last-commit/omadson/fuzzy-c-means.svg)](https://github.com/omadson/fuzzy-c-means/commit/master)
[![Downloads](https://pepy.tech/badge/fuzzy-c-means)](https://pepy.tech/project/fuzzy-c-means)
[![DOI](https://zenodo.org/badge/186457481.svg)](https://zenodo.org/badge/latestdoi/186457481)


`fuzzy-c-means` is a Python module implementing the [Fuzzy C-means][1] clustering algorithm.

## installation
the `fuzzy-c-means` package is available in [PyPI](https://pypi.org/project/fuzzy-c-means/). to install, simply type the following command:
```
pip install fuzzy-c-means
```
<!-- by default, the `fuzzy-c-means` uses [jax](https://github.com/google/jax) library, which only works on Linux systems. If you are a Windows user, try to install using:
```
pip install fuzzy-c-means[windows]
``` -->
### command line interface
<!-- if you prefer, you can install the command line interface so that you can use the library without having to program. To install, just use the command:
```
pip install fuzzy-c-means[cli]
``` -->
You can read the [CLI.md](https://github.com/omadson/fuzzy-c-means/blob/master/CLI.md) file for more information about this tool.

<!-- ## running examples

If you want to run the examples on [examples/](https://github.com/omadson/fuzzy-c-means/tree/master/examples) folder, try to install the extra dependencies.

```
pip install fuzzy-c-means[examples]
``` -->

### basic clustering example
simple example of use the `fuzzy-c-means` to cluster a dataset in two groups:

#### importing libraries
```Python
%matplotlib inline
import numpy as np
from fcmeans import FCM
from matplotlib import pyplot as plt
```

#### creating artificial data set
```Python
n_samples = 3000

X = np.concatenate((
    np.random.normal((-2, -2), size=(n_samples, 2)),
    np.random.normal((2, 2), size=(n_samples, 2))
))
```

#### fitting the fuzzy-c-means
```Python
fcm = FCM(n_clusters=2)
fcm.fit(X)
```

#### showing results
```Python
# outputs
fcm_centers = fcm.centers
fcm_labels = fcm.predict(X)

# plot result
f, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].scatter(X[:,0], X[:,1], alpha=.1)
axes[1].scatter(X[:,0], X[:,1], c=fcm_labels, alpha=.1)
axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
plt.savefig('images/basic-clustering-output.jpg')
plt.show()
```
<div align="center">
    <img src="https://raw.githubusercontent.com/omadson/fuzzy-c-means/master/examples/images/basic-clustering-output.jpg">
</div>

to more examples, see the [examples/](https://github.com/omadson/fuzzy-c-means/tree/master/examples) folder.


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
 - [Analisis Data Log IDS Snort dengan Algoritma Clustering Fuzzy C-Means](https://doi.org/10.24843/MITE.2020.v19i01.P14)
 - [Comparative Analysis between the k-means and Fuzzy c-means Algorithms to Detect UDP Flood DDoS Attack on a SDN/NFV Environment](https://doi.org/10.5220/0010176201050112)
 - [Mixture-of-Experts Variational Autoencoder for Clustering and Generating from Similarity-Based Representations on Single Cell Data](https://arxiv.org/abs/1910.07763)
 - [Fuzzy Clustering: an Application to Distributional Reinforcement Learning](https://doi.org/10.34726/hss.2021.86783)


## contributing and support

this project is open for contributions. here are some of the ways for you to contribute:
 - bug reports/fix
 - features requests
 - use-case demonstrations

please open an [issue](https://github.com/omadson/fuzzy-c-means/issues) with enough information for us to reproduce your problem. A [minimal, reproducible example](https://stackoverflow.com/help/minimal-reproducible-example) would be very helpful.

to make a contribution, just fork this repository, push the changes in your fork, open up an issue, and make a pull request!

## contributors
 - [Madson Dias](https://github.com/omadson)
 - [Dirk Nachbar](https://github.com/dirknbr)
 - [Alberth Florêncio](https://github.com/zealberth)

[1]: https://doi.org/10.1016/0098-3004(84)90020-7
[2]: http://scikit-learn.org/
