# fuzzy-c-means

![GitHub](https://img.shields.io/github/license/omadson/fuzzy-c-means.svg)
[![PyPI](https://img.shields.io/pypi/v/fuzzy-c-means.svg)](http://pypi.org/project/fuzzy-c-means/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/w/omadson/fuzzy-c-means.svg)](https://github.com/omadson/fuzzy-c-means/pulse)
[![GitHub last commit](https://img.shields.io/github/last-commit/omadson/fuzzy-c-means.svg)](https://github.com/omadson/fuzzy-c-means/commit/master)


`fuzzy-c-means` is a Python module implementing the [Fuzzy C-means][1] clustering algorithm.

## instalation
the `fuzzy-c-means` package is available in [PyPI](https://pypi.org/project/fuzzy-c-means/). to install, simply type the following command:
```
pip install fuzzy-c-means
```

## basic usage
simple example of use the `fuzzy-c-means` to cluster a dataset in tree groups:
```Python
from fcmeans import FCM
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from seaborn import scatterplot as scatter


# create artifitial dataset
n_samples = 50000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here
centers = [(-5, -5), (0, 0), (5, 5)]

X,_ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)

# fit the fuzzy-c-means
fcm = FCM(n_clusters=3)
fcm.fit(X)

# outputs
fcm_centers = fcm.centers
fcm_labels  = fcm.u.argmax(axis=1)


# plot result
%matplotlib inline
f, axes = plt.subplots(1, 2, figsize=(11,5))
scatter(X[:,0], X[:,1], ax=axes[0])
scatter(X[:,0], X[:,1], ax=axes[1], hue=fcm_labels)
scatter(fcm_centers[:,0], fcm_centers[:,1], ax=axes[1],marker="s",s=200)
plt.show()
```

## how to cite fuzzy-c-means package
if you use `fuzzy-c-means` package in your paper, please cite it in your publication.
```
@misc{fuzzy-c-means,
    author       = "Madson Luiz Dantas Dias",
    year         = "2019",
    title        = "fuzzy-c-means: An implementation of Fuzzy $C$-means clustering algorithm.",
    url          = "https://github.com/omadson/fuzzy-c-means",
    institution  = "Federal University of Cear\'{a}, Department of Computer Science" 
}
```

## contributing

this project is open for contributions. here are some of the ways for you to contribute:
 - bug reports/fix
 - features requests
 - use-case demonstrations

to make a contribution, just fork this repository, push the changes in your fork, open up an issue, and make a pull request!

## contributors
 - [Madson Dias](https://github.com/omadson)

[1]: https://doi.org/10.1016/0098-3004(84)90020-7
[2]: http://scikit-learn.org/





