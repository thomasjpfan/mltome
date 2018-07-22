from pathlib import Path

import pandas as pd
import pytest

from mltome.cache import from_dataframe_cache


@from_dataframe_cache('target')
def get_target(params, force=False, **kwargs):
    multiplier = params['multiplier']
    df = pd.DataFrame({"a": [1, 2, 3]})
    return multiplier * df


def test_dataframe_cache_unsupported_datatype(tmpdir):
    param = {'target': Path('hello.world'), 'multiplier': 2}
    with pytest.raises(ValueError):
        get_target(param)


def test_dataframe_cache_saves_to_cache(tmpdir):
    target = Path(tmpdir.mkdir('cache').join('output.fthr'))
    param = {'target': target, 'multiplier': 2}

    expected_df = pd.DataFrame({"a": [2, 4, 6]})
    output = get_target(param)
    assert output.equals(expected_df)
    assert target.exists()

    with target.open(mode='rb') as f:
        before = f.read()

    output = get_target(param)

    with target.open(mode='rb') as f:
        after = f.read()

    assert before == after


def test_dataframe_cache_updates_cache_with_force(tmpdir):
    target = Path(tmpdir.mkdir('cache').join('output2.fthr'))
    target.touch()
    previous_target_size = target.stat().st_size

    param = {'target': target, 'multiplier': 2}

    expected_df = pd.DataFrame({"a": [2, 4, 6]})
    output = get_target(param, force=True)
    assert output.equals(expected_df)
    assert target.exists()

    new_target_size = target.stat().st_size
    assert previous_target_size != new_target_size
