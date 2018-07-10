import logging

LOGGING_FORMAT = '%(message)s'


def get_log_file_handler(log_fn, level=logging.INFO):
    file_handler = logging.FileHandler(log_fn)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))
    return file_handler


def get_stream_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging.Formatter(LOGGING_FORMAT))

    logger.addHandler(stream_handler)

    return logger
