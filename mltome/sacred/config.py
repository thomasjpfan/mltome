import logging
import os
import datetime

from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

from .observers import CSVObserver, ArtifactObserver
from mltome.logging import get_stream_logger


def add_common_config(exp, csv_fn, record_local=True):
    exp.add_config(
        run_id=datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S'),
        record_local=record_local,
        name=exp.path)

    @exp.config
    def run_dir_config(name, run_id):
        model_id = f"{name}_{run_id}"  # noqa
        run_dir = os.path.join('artifacts', model_id)  # noqa

    exp.logger = get_stream_logger(exp.path)
    exp.observers.append(CSVObserver(csv_fn))
    exp.observers.append(ArtifactObserver(exp.logger))
    exp.captured_out_filter = apply_backspaces_and_linefeeds


def add_monogodb(observers, mongodb_url, mongodb_name):
    if mongodb_url and mongodb_name:
        observers.append(
            MongoObserver.create(url=mongodb_url, db_name=mongodb_name))


def add_pushover_handler(log, pushover_user_token, pushover_token):

    if pushover_user_token and pushover_token:
        from notifiers.logging import NotificationHandler
        h = NotificationHandler('pushover')
        h.setLevel(logging.WARNING)
        log.addHandler(h)


def add_neptune_observers(observers, use_neptune):
    if use_neptune:
        from mltome.neptune import NeptuneObserver
        observers.append(NeptuneObserver())
