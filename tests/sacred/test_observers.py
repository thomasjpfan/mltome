import os
import logging
import csv
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from mltome.sacred.observers import (CSVObserver, ArtifactObserver,
                                     PushoverObserver)


@pytest.fixture
def result_fn(tmpdir):
    return Path(tmpdir.join('results.csv'))


@pytest.fixture
def log():
    return logging.getLogger('obs')


def test_csvobserver_existing_result_fn(result_fn):
    result_fn.touch()
    before = result_fn.stat().st_size
    CSVObserver(result_fn)
    after = result_fn.stat().st_size

    # Did not change
    assert before == after


def test_csvobserver_new_result_fn(result_fn):
    CSVObserver(result_fn)

    assert result_fn.exists()


def test_csvobserver_non_train_command_started_event(result_fn):
    obs = CSVObserver(result_fn)

    obs.started_event(None, 'predict', None, None, None, None, None)

    assert not obs.record_local

    obs.completed_event(1, [1, 2, 3])


def test_csvobserver_no_model_id_or_record_local(result_fn):
    obs = CSVObserver(result_fn)

    obs.started_event(None, 'train', None, None, {}, None, 'a_id')

    assert obs.model_id == 'a_id_train'
    assert not obs.record_local


def test_csvobserver_with_model_id_and_record(result_fn):
    obs = CSVObserver(result_fn)

    obs.started_event(None, 'train', None, 1, {
        'model_id': 'id1',
        'record_local': True
    }, None, 'a_id')

    assert obs.model_id == 'id1_train'
    assert obs.record_local
    assert obs.start_time == 1


def test_csvobserver_cycle(result_fn):
    obs = CSVObserver(result_fn)

    obs.started_event(None, 'train', None, datetime.now(), {
        'model_id': 'id1',
        'record_local': True
    }, None, 'a_id')

    obs.completed_event(datetime.now(), [1, 2])

    rows = []
    with result_fn.open('r') as f:
        reader = csv.DictReader(f, fieldnames=obs.COLS)
        next(reader)
        for row in reader:
            rows.append(row)

    assert len(rows) == 1

    # new id
    obs.started_event(None, 'train', None, datetime.now(), {
        'model_id': 'id2',
        'record_local': True
    }, None, 'b_id')
    obs.completed_event(datetime.now(), [1, 2])

    rows = []
    with result_fn.open('r') as f:
        reader = csv.DictReader(f, fieldnames=obs.COLS)
        next(reader)
        for row in reader:
            rows.append(row)

    assert len(rows) == 2

    # Same as first id
    obs.started_event(None, 'train', None, datetime.now(), {
        'model_id': 'id1',
        'record_local': True
    }, None, 'c_id')

    obs.completed_event(datetime.now(), [1, 2])

    rows = []
    with result_fn.open('r') as f:
        reader = csv.DictReader(f, fieldnames=obs.COLS)
        next(reader)
        for row in reader:
            rows.append(row)

    assert len(rows) == 2
    assert rows[-1]['model_id'] == 'id1_train'


def test_artifactobserver_no_run_dir(log):
    ao = ArtifactObserver(log)

    with pytest.raises(EnvironmentError):
        ao.started_event(None, None, None, None, {}, None, None)


def test_artifactobserver_predict_no_dir(log, tmpdir):
    a_dir = str(tmpdir.join('obs'))
    ao = ArtifactObserver(log)

    with pytest.raises(EnvironmentError):
        ao.started_event(None, 'predict', None, None, {'run_dir': str(a_dir)},
                         None, None)


def test_artifactobserver_train_with_dir(log, tmpdir):
    a_dir = str(tmpdir.join('obs'))
    ao = ArtifactObserver(log)

    ao.started_event(None, 'train', None, None, {'run_dir': str(a_dir)}, None,
                     None)

    assert log.handlers
    assert ao.val_test_score_fn == os.path.join(a_dir, 'val_train_score.txt')

    ao.completed_event(None, [1, 2])
    assert os.path.exists(ao.val_test_score_fn)


def test_pushoverobserver(monkeypatch):
    pushover_mock = Mock()
    po = PushoverObserver(pushover_mock, 'tf', '123')

    po.completed_event(None, [0.5, 0.3])
    msg = 'Valid score: 0.5, Train score: 0.3'
    pushover_mock.notify.assert_called_once_with(
        message=msg, user='tf', token='123')
