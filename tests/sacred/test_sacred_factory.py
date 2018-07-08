import yaml

from mltome.sacred import generate_experiment_params_from_env


def test_generate_experiment_params_from_env(tmpdir):
    root_dir = tmpdir.mkdir('root')
    config_fn = root_dir.join('neptune.yml')
    csv_fn = str(tmpdir.mkdir('artifacts').join('results.csv'))

    config = {'parameters': {'hello': 'world'}}

    with open(config_fn, 'w') as f:
        yaml.dump(config, f)

    exp, params = generate_experiment_params_from_env(
        'test',
        tags=['test_tag'],
        csv_fn=csv_fn,
        root_dir=str(root_dir),
        config_fn='neptune.yml')

    assert params['hello'] == 'world'
