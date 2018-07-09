
"""Top-level package for mltome."""
import os
from pathlib import Path
import yaml
from munch import munchify

from skorch.callbacks import EpochScoring
from skorch.callbacks import Checkpoint

from .neptune import NeptuneSkorchCallback
from .skorch.callbacks import LRRecorder, HistorySaver, TensorboardXLogger


def get_params(root_dir=".",
               config_fn="neptune.yaml",
               raw_root="data/raw",
               process_root="data/proc"):
    config_fn = os.path.join(root_dir, config_fn)
    with open(config_fn, "r") as f:
        config = yaml.safe_load(f)

    params = config['parameters']
    for key, value in params.items():
        if key.startswith("files__raw_"):
            config['parameters'][key] = Path(
                os.path.join(root_dir, raw_root, value))
        elif key.startswith("files__proc_"):
            config['parameters'][key] = Path(
                os.path.join(root_dir, process_root, value))

    return munchify(params)


def get_neptune_skorch_callback(batch_targets=None, epoch_targets=None):
    use_neptune = os.environ.get('USE_NEPTUNE')

    if use_neptune != 'true':
        return None

    return NeptuneSkorchCallback(batch_targets=batch_targets,
                                 epoch_targets=epoch_targets)


def get_classification_skorch_callbacks(
        model_id, checkpoint_fn, history_fn,
        pgroups, artifacts_dir='artifacts/run', per_epoch=True):

    pgroup_names = [item[0] + "_lr" for item in pgroups]
    tensorboard_log_dir = os.path.join(artifacts_dir, model_id)

    batch_targets = ['train_loss']
    epoch_targets = ['train_acc', 'valid_acc']
    if per_epoch:
        epoch_targets.extend(pgroup_names)
    else:
        batch_targets.extend(pgroup_names)

    callbacks = [
        EpochScoring(
            'accuracy', name='train_acc', lower_is_better=False,
            on_train=True),
        LRRecorder(group_names=pgroup_names),
        TensorboardXLogger(
            tensorboard_log_dir,
            batch_targets=batch_targets,
            epoch_targets=epoch_targets,
            epoch_groups=['acc']),
        Checkpoint(target=checkpoint_fn),
        HistorySaver(target=history_fn)
    ]

    neptune_callback = get_neptune_skorch_callback(
        batch_targets=batch_targets, epoch_targets=epoch_targets)
    if neptune_callback is not None:
        callbacks.append(neptune_callback)

    return callbacks
