import yaml

from mltome import get_params, get_classification_skorch_callbacks


def test_get_params(tmpdir):
    root_dir = tmpdir.mkdir('root')
    config_fn = root_dir.join('neptune.yml')

    config = {
        'parameters': {
            'hello': 'world',
            'files__raw_first': 'raw_file.csv',
            'files__proc_another': 'proc_file.csv'
        }
    }

    with open(config_fn, 'w') as f:
        yaml.dump(config, f)

    params = get_params(root_dir=str(root_dir), config_fn=str(config_fn))

    assert str(
        params.files__raw_first) == root_dir.join('data/raw/raw_file.csv')
    assert str(
        params.files__proc_another) == root_dir.join('data/proc/proc_file.csv')
    assert params.hello == 'world'


def test_get_classification_skorch_callbacks():
    callbacks = get_classification_skorch_callbacks('test', 'checkpoint_fn',
                                                    'history_fn', [])
    assert len(callbacks) == 5
