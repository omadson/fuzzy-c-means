import numpy as np


class _ArrayMeta(type):
    def __getitem__(self, t):
        return type("ArrayLike", (ArrayLike,), {"__dtype__": t})


class ArrayLike(np.ndarray, metaclass=_ArrayMeta):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        if isinstance(val, np.ndarray):
            return val
        raise ValueError(f"{val} is not an instance of numpy.ndarray")
