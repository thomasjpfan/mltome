from pathlib import Path

import pandas as pd

from mltome.cache import from_dataframe_cache


def get_target():
    @from_dataframe_cache('target')
    def output(params, force=False, **kwargs):
        multiplier = params['multiplier']
        df = pd.DataFrame({"a": [1, 2, 3]})
        return multiplier * df

    return output


def test_dataframe_cache_saves_to_cache(tmpdir):
    target = Path(tmpdir.mkdir('cache').join('output.parq'))
    param = {'target': target, 'multiplier': 2}

    expected_df = pd.DataFrame({"a": [2, 4, 6]})
    output = get_target(param)
    assert output.equals(expected_df)
    assert target.exists()

    # before_mtime = target.stat().st_mtime
    # output = get_target(param)
    # after_mtime = target.stat().st_mtime
    # assert before_mtime == after_mtime


# def test_dataframe_cache_updates_cache_with_force(tmpdir, get_target):
    # target = Path(tmpdir.mkdir('cache').join('output.parq'))
    # param = {'target': target, 'multiplier': 2}

    # expected_df = pd.DataFrame({"a": [2, 4, 6]})
    # output = get_target(param)
    # assert output.equals(expected_df)
    # assert target.exists()

    # before_mtime = target.stat().st_mtime
    # output = get_target(param, force=True)
    # after_mtime = target.stat().st_mtime
    # assert before_mtime != after_mtime
