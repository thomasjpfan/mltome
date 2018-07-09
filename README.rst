mltome
======

A collection of Mtools that builds on Python ML frameworks. The API of these tools are **not** stable!

.. image:: https://circleci.com/gh/thomasjpfan/mltome.svg?style=shield
    :target: https://circleci.com/gh/thomasjpfan/mltome
    :alt: CI Status

.. image:: https://codecov.io/gh/thomasjpfan/mltome/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/thomasjpfan/mltome
    :alt: Codecov Status


Installation
------------

You can install mltome directly from pypi:

.. code-block:: bash

    pip install git+https://github.com/thomasjpfan/mltome

- ``mltome.pytorch`` - For installation
instructions for PyTorch, visit the `PyTorch website
<http://pytorch.org/>`__.

Additional Installation
-----------------------

- ``mltome.skorch.TensorboardXLogger`` - ``pip install tensorboardX``
- ``mltome.neptune.NeptuneSkorchCallback`` - ``pip install neptune-cli``
- ``mltome.skorch.callbacks.HistorySaver`` - ``pip install git+https://github.com/dnouri/skorch@24ac0d1392306da2337174eba206446fab7b179c``
- ``mltome.config.add_pushover_handler`` - ``pip install notifiers``
- ``mltome.config.add_monogodb`` - ``pip install pymongo``

Usage
-----

``mltome.sacred.generate_experiment_params_from_env`` is used to generate a skorch experiment
It uses the following env variables:

- ``MONGODB_NAME`` and ``MONGODB_URL`` for the database name and the url for a mongodb server to be used with sacred.
- ``NOTIFIERS_PUSHOVER_USER`` and ``NOTIFIERS_PUSHOVER_TOKEN`` to push notifications to pushover.
- Setting ``USE_NEPTUNE`` equal to ``true`` will send stats to `neptune.ml <https://neptune.ml>`_.
- ``mltome.get_neptune_skorch_callback`` also uses ``USE_NEPTUNE`` to configure a skorch callback for deep learning.


Development
-----------

The development version can be installed by running ``make dev``. Then we can lint ``make lint`` and tests by running ``pytest``.
