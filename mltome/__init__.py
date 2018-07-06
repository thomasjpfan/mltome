"""Top-level package for mltome."""
import os
from pathlib import Path
import yaml
from munch import munchify


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

