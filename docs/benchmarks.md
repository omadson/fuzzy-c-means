# Benchmarks

Time and memory of `fit`, measured with
[`benchmarks/benchmark.py`](https://github.com/omadson/fuzzy-c-means/blob/master/benchmarks/benchmark.py).
To reproduce them on your machine:

```bash
uv run python benchmarks/benchmark.py
```

The numbers below come from a single laptop (Intel Core i7-10510U, 4
cores / 8 threads, 15 GB RAM, Linux, Python 3.12, numpy 1.26) and are
meant to show how cost scales, not to be absolute figures.

How it is measured:

- Every fit runs exactly 20 iterations (`max_iter=20`, `error=1e-9`) on
  uniform noise, which never meets the stopping rule early, so times are
  comparable across sizes. A fit that converges sooner costs
  proportionally less.
- Time is the best of 3 runs. Memory is the peak of `tracemalloc` during
  `fit`: it counts the buffers the algorithm allocates, not the input
  array.

## Samples

FCM with $c=5$ clusters and $p=10$ features.

| n | time (s) | peak memory (MB) |
| --- | --- | --- |
| 1,000 | 0.01 | 1 |
| 10,000 | 0.08 | 5 |
| 100,000 | 1.89 | 50 |
| 500,000 | 7.82 | 248 |

Both grow linearly with $n$. The peak is dominated by the
$n \times c \times p$ array of differences that the distance computation
builds, about $8ncp$ bytes (190 MB at $n=500{,}000$) plus the
$n \times c$ partition matrix. The whole data set must fit in memory
together with that array: there is no batch or out-of-core mode yet.

## Clusters and features

FCM with $n=20{,}000$.

| c | p | time (s) | peak memory (MB) |
| --- | --- | --- | --- |
| 2 | 2 | 0.09 | 2 |
| 2 | 10 | 0.25 | 4 |
| 2 | 50 | 0.42 | 16 |
| 5 | 2 | 0.13 | 5 |
| 5 | 10 | 0.29 | 10 |
| 5 | 50 | 0.59 | 40 |
| 20 | 2 | 0.33 | 18 |
| 20 | 10 | 0.74 | 40 |
| 20 | 50 | 1.32 | 162 |

Memory is proportional to $c \cdot p$, as above. Time grows more slowly
than $c \cdot p$ at small sizes, because part of the cost is fixed per
iteration.

## Variants

All models on the same data: $n=2{,}000$, $c=5$, $p=10$.

| model | time (s) | peak memory (MB) |
| --- | --- | --- |
| FCM | 0.03 | 1.0 |
| PCM | 0.03 | 1.0 |
| FPCM | 0.03 | 1.1 |
| GK | 0.35 | 1.3 |
| GG | 0.81 | 1.4 |
| KFCM | 0.03 | 1.1 |
| FCMedoids | 0.22 | 74.0 |

- `PCM`, `FPCM` and `KFCM` cost about the same as `FCM`.
- `GK` and `GG` are 10 to 30 times slower: each iteration builds and
  inverts one $p \times p$ covariance matrix per cluster. The cost grows
  with $p^3$, so they are best kept to a moderate number of features.
- `FCMedoids` stores the $n \times n$ distance matrix, so its memory grows
  with $n^2$ whatever $c$ and $p$ are:

    | n | time (s) | peak memory (MB) | n² float64 (MB) |
    | --- | --- | --- | --- |
    | 1,000 | 0.08 | 33 | 8 |
    | 2,000 | 0.27 | 74 | 31 |
    | 4,000 | 0.86 | 244 | 122 |
    | 8,000 | 6.52 | 977 | 488 |

    The peak is about twice the matrix. Beyond a few thousand samples use
    another variant.

## Initialization

Quality of the optimum reached, over 40 seeds, on 8 overlapping Gaussian
blobs (1,200 points, $c=8$). "Runs at the best optimum" is the share of
fits whose objective is within 0.1% of the lowest value found by any
configuration.

| init | n_init | mean objective | std | runs at the best optimum | time per fit (s) |
| --- | --- | --- | --- | --- | --- |
| random | 1 | 966.3 | 16.6 | 85% | 0.029 |
| k-means++ | 1 | 967.5 | 17.6 | 82% | 0.028 |
| random | 5 | 959.3 | 0.0 | 100% | 0.150 |
| k-means++ | 5 | 959.3 | 0.0 | 100% | 0.225 |

- `n_init` is what matters here: 5 restarts removed the bad optima in all
  40 seeds, at roughly 5 times the cost.
- `init="k-means++"` made no measurable difference on this data. The
  fuzzy partition that `"random"` starts from is already close to uniform,
  so it rarely lands in a poor basin. It may help more on data with many
  tight, uneven clusters, so measure it on yours before relying on it.
