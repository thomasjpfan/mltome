import pytest
from pathlib import Path

import pandas as pd

from mltome.cache import from_dataframe_cache


@pytest.fixture
def target(tmpdir):
    return Path(tmpdir.mkdir('cache').join('output.parquet'))


@pytest.fixture
def param(target):
    return {'target': target, 'multiplier': 2}


@pytest.fixture
def get_target():

    @from_dataframe_cache('target')
    def get_target(params, force=False, **kwargs):
        multiplier = params['multiplier']
        df = pd.DataFrame({"a": [1, 2, 3]})
        return multiplier * df

    return get_target


def test_dataframe_cache_saves_to_cache(param, target, get_target):

    expected_df = pd.DataFrame({"a": [2, 4, 6]})
    output = get_target(param)
    assert output.equals(expected_df)
    assert target.exists()

    before_mtime = target.stat().st_mtime
    output = get_target(param)
    after_mtime = target.stat().st_mtime
    assert before_mtime == after_mtime


def test_dataframe_cache_updates_cache_with_force(param, target, get_target):
    expected_df = pd.DataFrame({"a": [2, 4, 6]})
    output = get_target(param)
    assert output.equals(expected_df)
    assert target.exists()

    before_mtime = target.stat().st_mtime
    output = get_target(param, force=True)
    after_mtime = target.stat().st_mtime
    assert before_mtime != after_mtime
