import logging

from mltome.logging import get_log_file_handler
from mltome.logging import get_stream_logger


def test_get_log_file_handler(tmpdir):
    log_fn = tmpdir.mkdir('log').join('file.log')
    fh = get_log_file_handler(log_fn)

    assert isinstance(fh, logging.FileHandler)


def test_get_stream_logger():
    sl = get_stream_logger('hello')

    level = sl.getEffectiveLevel()
    assert level == logging.INFO
