"""Time and memory benchmarks of fcmeans, printed as Markdown tables.

Run with `uv run python benchmarks/benchmark.py`. Every timed fit runs
exactly `ITER` iterations on uniform noise (which never converges early),
so times are comparable across sizes. Memory is the peak of `tracemalloc`
during `fit`, which tracks numpy buffers but not the input array itself.
"""

import platform
import time
import tracemalloc

import numpy as np

from fcmeans import FCM, FPCM, GG, GK, KFCM, PCM, FCMedoids

ITER = 20
REPEAT = 3


def make(n, p, seed=0):
    """Uniform noise: the stopping rule never fires before `ITER`."""
    return np.random.default_rng(seed).uniform(size=(n, p))


def measure(cls, X, **kw):
    """Best-of-REPEAT fit time (s) and peak memory (MB) of one fit."""
    kw = dict(max_iter=ITER, error=1e-9, random_state=0, **kw)
    times = []
    for _ in range(REPEAT):
        t = time.perf_counter()
        cls(**kw).fit(X)
        times.append(time.perf_counter() - t)
    tracemalloc.start()
    cls(**kw).fit(X)
    peak = tracemalloc.get_traced_memory()[1] / 2**20
    tracemalloc.stop()
    return min(times), peak


def table(header, rows):
    """Print a Markdown table."""
    print("| " + " | ".join(header) + " |")
    print("|" + " --- |" * len(header))
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def scaling_n():
    """FCM against the number of samples."""
    print(f"### Samples (FCM, c=5, p=10, {ITER} iterations)\n")
    rows = []
    for n in (1_000, 10_000, 100_000, 500_000):
        t, mem = measure(FCM, make(n, 10), n_clusters=5)
        rows.append((f"{n:,}", f"{t:.2f}", f"{mem:.0f}"))
    table(["n", "time (s)", "peak memory (MB)"], rows)


def scaling_cp():
    """FCM against clusters and features."""
    print(f"### Clusters and features (FCM, n=20,000, {ITER} iterations)\n")
    rows = []
    for c in (2, 5, 20):
        for p in (2, 10, 50):
            t, mem = measure(FCM, make(20_000, p), n_clusters=c)
            rows.append((c, p, f"{t:.2f}", f"{mem:.0f}"))
    table(["c", "p", "time (s)", "peak memory (MB)"], rows)


def variants():
    """Every variant on the same data."""
    print(f"### Variants (n=2,000, c=5, p=10, {ITER} iterations)\n")
    rows = []
    for cls in (FCM, PCM, FPCM, GK, GG, KFCM, FCMedoids):
        t, mem = measure(cls, make(2_000, 10), n_clusters=5)
        rows.append((cls.__name__, f"{t:.2f}", f"{mem:.1f}"))
    table(["model", "time (s)", "peak memory (MB)"], rows)


def medoids_n():
    """FCMedoids builds an n x n distance matrix."""
    print(f"### FCMedoids memory (c=5, p=10, {ITER} iterations)\n")
    rows = []
    for n in (1_000, 2_000, 4_000, 8_000):
        t, mem = measure(FCMedoids, make(n, 10), n_clusters=5)
        rows.append(
            (f"{n:,}", f"{t:.2f}", f"{mem:.0f}", f"{n * n * 8 / 2**20:.0f}")
        )
    table(["n", "time (s)", "peak memory (MB)", "n² float64 (MB)"], rows)


def objective(model, X):
    """FCM objective of a fitted model."""
    return (model.u**model.m * model._distances(X) ** 2).sum()


def initialization():
    """Quality of the optimum reached, by `init` and `n_init`."""
    rng = np.random.default_rng(1)
    centers = rng.uniform(0, 10, size=(8, 2))
    X = np.vstack([c + rng.normal(size=(150, 2)) for c in centers])
    seeds = range(40)
    runs = {}
    for init, n_init in (
        ("random", 1),
        ("k-means++", 1),
        ("random", 5),
        ("k-means++", 5),
    ):
        js, t = [], time.perf_counter()
        for s in seeds:
            fcm = FCM(n_clusters=8, init=init, n_init=n_init, random_state=s)
            fcm.fit(X)
            js.append(objective(fcm, X))
        runs[(init, n_init)] = (np.array(js), time.perf_counter() - t)
    best = min(js.min() for js, _ in runs.values())
    print("### Initialization (8 overlapping blobs, 40 seeds)\n")
    rows = []
    for (init, n_init), (js, t) in runs.items():
        hit = np.mean(js <= best * (1 + 1e-3))
        rows.append(
            (
                init,
                n_init,
                f"{js.mean():.1f}",
                f"{js.std():.1f}",
                f"{hit:.0%}",
                f"{t / len(seeds):.3f}",
            )
        )
    table(
        [
            "init",
            "n_init",
            "mean objective",
            "std",
            "runs at the best optimum",
            "time per fit (s)",
        ],
        rows,
    )


if __name__ == "__main__":
    print(
        f"Python {platform.python_version()}, numpy {np.__version__}, "
        f"{platform.processor() or platform.machine()}\n"
    )
    for step in (scaling_n, scaling_cp, variants, medoids_n, initialization):
        step()
