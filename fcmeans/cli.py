"""
Fuzzy C-means

Fit and use fuzzy-c-means models to clustering.

You probably want to install completion for the typer command.
If you are using bash, try to type:

$ fcm --install-completion bash

https://github.com/omadson/fuzzy-c-means
"""
import pickle
import time
from pathlib import Path

import numpy as np
import typer
from tabulate import tabulate

from . import FCM

app = typer.Typer(help=__doc__)


class Options:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def extension_check(extension: str, value: Path):
    if value.suffix != extension:
        raise typer.BadParameter(f"File '{value}' must be extension '{extension}'.")
    return value


def input_path_callback(value: Path):
    return extension_check(".csv", value)


def model_path_callback(value: Path):
    if value.exists():
        typer.confirm(
            f"Do you confirm to replace '{value}' file?",
            default=True,
            abort=True,
        )
    return value


def delimiter_callback(value: str):
    delimiters = [" ", ",", "|", ";"]
    if value in delimiters:
        return value
    raise typer.BadParameter(
        f"The delimiters must be in the following list: {delimiters}."
    )


def _predict(data, model):
    return model.predict(data)


def _read_data(dataset_path, delimiter, quiet):
    typer.echo()
    if not quiet:
        typer.echo("Reading data set... ", nl=False)
    X = np.genfromtxt(dataset_path, delimiter=delimiter)
    # Check file read
    if not np.all(X):
        typer.echo(
            "Error: Please verify if value for '--delimiter' / '-d' is "
            f"the delimiter of the '{dataset_path}' file."
        )
        raise typer.Abort()
    if np.isnan(np.sum(X)):
        typer.echo(f"Error: File '{dataset_path}' cannot contain NaN.")
        raise typer.Abort()
    if not quiet:
        typer.echo("Data set read without errors...")
    return X


def _model_predict(model, X, dataset_path, delimiter, quiet):
    labels = model.predict(X)
    new_file_name = dataset_path.with_suffix(".labels.csv")
    if new_file_name.exists():
        typer.confirm(
            f"Do you confirm to replace '{new_file_name}' file?",
            default=True,
            abort=True,
        )
    np.savetxt(new_file_name, labels, delimiter=delimiter, fmt="%d")
    if not quiet:
        typer.echo(f"Model predictions has been saved as '{new_file_name}'.")


@app.command()
def fit(
    dataset_path: Path = typer.Argument(
        "dataset.csv",
        help="Data set file path (only .csv).",
        dir_okay=False,
        exists=True,
        callback=input_path_callback,
    ),
    delimiter: str = typer.Option(
        ",",
        "--delimiter",
        "-d",
        help="Delimiter of data set file.",
        callback=delimiter_callback,
    ),
    model_path: Path = typer.Argument(
        "model.sav",
        help="Path to save the created model.",
        dir_okay=False,
        callback=model_path_callback,
    ),
    n_clusters: int = typer.Option(
        2, "--clusters", "-c", max=500, help="Number of clusters."
    ),
    m: float = typer.Option(
        2.0,
        "--exponent",
        "-e",
        min=1,
        max=100,
        help="Fuzzy partition exponent.",
    ),
    max_iter: int = typer.Option(
        150,
        "--max-iter",
        "-m",
        min=1,
        max=5000,
        help="Maximum number of iterations.",
    ),
    error: float = typer.Option(
        1e-5, "--tolerance", "-t", min=1e-10, help="Stop Tolerance criteria."
    ),
    random_state: int = typer.Option(
        None, "--seed", "-s", help="Seed for the random number generator."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress model info."),
    predict: bool = typer.Option(False, "--predict", "-p", help="Prediction flag."),
):
    """Train and save a fuzzy-c-means model given a dataset."""
    X = _read_data(dataset_path, delimiter, quiet)
    model = FCM(
        n_clusters=n_clusters,
        max_iter=max_iter,
        m=m,
        error=error,
        random_state=random_state,
    )
    if not quiet:
        typer.echo("Training model... ", nl=False)
    start_time = time.time()
    try:
        model.fit(X)
    except Exception:
        typer.echo("\nError: There was an error in the fitting step.")
        typer.echo(
            "If the problem continues, create an issue at:"
            "https://github.com/omadson/fuzzy-c-means/issues"
        )
        raise typer.Abort()
    if not quiet:
        typer.echo("Model trained without errors...")
    elapsed_time = (time.time() - start_time) * 1000
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    if not quiet:
        headers = ["Variable", "Value"]
        table = [
            ["Number of clusters", n_clusters],
            ["Fuzzy partition matrix exponent", m],
            ["Stop tolerance criteria", error],
            ["Training time (ms)", elapsed_time],
        ]
        typer.echo("\nModel info:")
        typer.echo(tabulate(table, headers, tablefmt="fancy_grid"))
    if not quiet:
        typer.echo(f"\nYour model has been saved as '{model_path}'.")
    if predict:
        _model_predict(model, X, dataset_path, delimiter, quiet)


@app.command()
def predict(
    dataset_path: Path = typer.Argument(
        "dataset.csv",
        help="Data set file path (only .csv).",
        dir_okay=False,
        exists=True,
        callback=input_path_callback,
    ),
    delimiter: str = typer.Option(
        ",",
        "--delimiter",
        "-d",
        help="Delimiter of data set file.",
        callback=delimiter_callback,
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress model info."),
    model_path: Path = typer.Argument(
        "model.sav",
        help="Path to save the created model.",
        dir_okay=False,
        exists=True,
    ),
):
    """Predict labels given a data set and a fuzzy-c-means saved model."""
    X = _read_data(dataset_path, delimiter, quiet)
    if not quiet:
        typer.echo("Reading model... ", nl=False)
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
    except Exception:
        typer.echo("Error: Something wrong with your models.")
        raise typer.Abort()
    if not quiet:
        typer.echo("Model loaded without errors...")
    _model_predict(model, X, dataset_path, delimiter, quiet)
