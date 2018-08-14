import logging
from sacred import Experiment

from mltome.sacred.config import (add_monogodb, add_pushover_handler,
                                  add_neptune_observers)


def test_add_monogodb():
    obs = []
    add_monogodb(obs, 'url', 'name')
    assert len(obs) == 1


def test_add_pushover_handler():
    log = logging.getLogger('config')

    assert not log.handlers
    add_pushover_handler(log, 'push_user', 'token')

    assert log.handlers


def test_neptune_observers():
    obs = []
    add_neptune_observers(obs, 'model_id', 'tags', True)
    assert obs
