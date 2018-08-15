mltome
======

A collection of ML tools that builds on Python ML frameworks. The API of these tools are **not** stable!

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
- ``mltome.config.add_pushover_observer`` - ``pip install notifiers``
- ``mltome.config.add_monogodb`` - ``pip install pymongo``

Development
-----------

The development version can be installed by running ``make dev``. Then we can lint ``make lint`` and tests by running ``pytest``.
