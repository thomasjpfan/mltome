"""Caching utilities"""
from functools import wraps

import feather


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
                return feather.read_dataframe(fn)
            output = f(params, force, **kwargs)
            feather.write_dataframe(output, fn)
            return output

        return wrapper

    return cache_decorator
