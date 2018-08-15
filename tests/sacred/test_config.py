from mltome.sacred.config import (add_monogodb, add_pushover_observer,
                                  add_neptune_observers)


def test_add_monogodb():
    obs = []
    add_monogodb(obs, 'url', 'name')
    assert len(obs) == 1


def test_add_pushover_observer():
    obs = []
    add_pushover_observer(obs, 'push_user', 'token')
    assert obs


def test_neptune_observers():
    obs = []
    add_neptune_observers(obs, 'model_id', 'tags', True)
    assert obs
