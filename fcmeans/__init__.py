# flake8: noqa
"""fuzzy-c-means - A simple implementation of Fuzzy C-means algorithm."""
from .fcmedoids import FCMedoids
from .fpcm import FPCM
from .gg import GG
from .gk import GK
from .kfcm import KFCM
from .main import FCM
from .pcm import PCM
# fmt: off
from .validation import (davies_bouldin, fukuyama_sugeno, fuzzy_silhouette,
                         select_n_clusters, xie_beni)

__version__ = "2.1.0"
