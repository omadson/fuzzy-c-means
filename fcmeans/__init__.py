"""fuzzy-c-means - A simple implementation of Fuzzy C-means algorithm."""
import platform

if platform.system() in ['Linux', 'Darwin']:
    from ._jax import FCM
else:
    from ._numpy import FCM
