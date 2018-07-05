"""Caching utilities"""
from functools import wraps

import pandas as pd


def from_dataframe_cache(key):
    """Returns a decorator that wraps a function with signature,
    (params: dict, force: bool, kwargs) that returns a pandas
    Dataframe.

    The ``params[key]`` should be a :class:`pathlib.Path` to
    save and load the cached dataframe from.

    Parameters
    ----------
    key: str
        key to query ``params`` to get path of cache

    """

    def cache_decorator(f):
        @wraps(f)
        def wrapper(params, force=False, **kwargs):
            fn = params[key]
            if fn.exists() and not force:
                return pd.read_parquet(fn)
            output = f(params, force, **kwargs)
            output.to_parquet(fn)
            return output

        return wrapper

    return cache_decorator
