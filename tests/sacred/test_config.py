from sacred import Experiment

from mltome.sacred.config import add_common_config


def test_add_common_config(tmpdir):
    csv_fn = str(tmpdir.mkdir('artifacts').join('results.csv'))

    exp = Experiment("test")

    add_common_config(exp, csv_fn)

    assert len(exp.observers) == 2
