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
            is_feather = fn.suffix in [".fthr"]
            is_parq = fn.suffix in [".parq"]

            if not is_feather and not is_parq:
                raise ValueError(f"Unsupported data type: {fn}")

            if fn.exists() and not force:
                if is_feather:
                    return pd.read_feather(fn)
                else:
                    return pd.read_parquet(fn)
            output = f(params, force, **kwargs)
            if is_feather:
                output.to_feather(fn)
            else:
                output.to_parquet(fn)
            return output

        return wrapper

    return cache_decorator
