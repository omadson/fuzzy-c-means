# `fcm`

Fuzzy C-means

Fit and use fuzzy-c-means models to clustering.

You probably want to install completion for the typer command.
If you are using bash, try to type:

$ fcm --install-completion bash

https://github.com/omadson/fuzzy-c-means

**Usage**:

```console
$ fcm [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `fit`: Train and save a fuzzy-c-means model given a...
* `predict`: Predict labels given a data set and a...

## `fcm fit`

Train and save a fuzzy-c-means model given a dataset.

**Usage**:

```console
$ fcm fit [OPTIONS] [DATASET_PATH] [MODEL_PATH]
```

**Arguments**:

* `[DATASET_PATH]`: Data set file path (only .csv).  [default: dataset.csv]
* `[MODEL_PATH]`: Path to save the created model.  [default: model.sav]

**Options**:

* `-d, --delimiter TEXT`: Delimiter of data set file.  [default: ,]
* `-c, --clusters INTEGER RANGE`: Number of clusters.  [default: 2]
* `-e, --exponent FLOAT RANGE`: Fuzzy partition exponent.  [default: 2.0]
* `-m, --max-iter INTEGER RANGE`: Maximum number of iterations.  [default: 150]
* `-t, --tolerance FLOAT RANGE`: Stop Tolerance criteria.  [default: 1e-05]
* `-s, --seed INTEGER`: Seed for the random number generator.
* `-q, --quiet`: Suppress model info.  [default: False]
* `-p, --predict`: Prediction flag.  [default: False]
* `--help`: Show this message and exit.

## `fcm predict`

Predict labels given a data set and a fuzzy-c-means saved model.

**Usage**:

```console
$ fcm predict [OPTIONS] [DATASET_PATH] [MODEL_PATH]
```

**Arguments**:

* `[DATASET_PATH]`: Data set file path (only .csv).  [default: dataset.csv]
* `[MODEL_PATH]`: Path to save the created model.  [default: model.sav]

**Options**:

* `-d, --delimiter TEXT`: Delimiter of data set file.  [default: ,]
* `-q, --quiet`: Suppress model info.  [default: False]
* `--help`: Show this message and exit.
