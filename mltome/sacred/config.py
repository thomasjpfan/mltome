import logging


def add_monogodb(observers, mongodb_url, mongodb_name):
    if mongodb_url and mongodb_name:
        from sacred.observers import MongoObserver
        observers.append(
            MongoObserver.create(url=mongodb_url, db_name=mongodb_name))


def add_pushover_handler(log, pushover_user_token, pushover_token):

    if pushover_user_token and pushover_token:
        from notifiers.logging import NotificationHandler
        h = NotificationHandler('pushover')
        h.setLevel(logging.WARNING)
        log.addHandler(h)


def add_neptune_observers(observers, model_id_key, tags_key, ctx=None):
    if ctx is not None:
        from mltome.neptune import NeptuneObserver
        observers.append(NeptuneObserver(ctx, model_id_key, tags_key))
