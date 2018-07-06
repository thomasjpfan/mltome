import pytest
import numpy as np
import torch
import torch.nn as nn

from skorch.net import NeuralNet
from mltome.skorch.callbacks import LRRecorder


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        return self.layer2(self.layer1(x))


@pytest.fixture
def data():
    X = np.ones((1000, 1), dtype='float32')
    y = np.ones((1000, 1), dtype='float32')
    return X, y


# LRRecorder
def test_lr_recorder_default_lr_per_epoch(data):
    module = MyModule()
    X, y = data

    recorder = LRRecorder(per_epoch=True)
    net = NeuralNet(
        module,
        torch.nn.MSELoss,
        callbacks=[recorder],
        max_epochs=5,
        optimizer__lr=0.1,
        batch_size=100)
    net.fit(X, y)

    lrs = net.history[:, 'default_lr']
    assert lrs == [0.1] * 5


def test_lr_recorder_default_lr_per_batch(data):
    module = MyModule()
    X, y = data

    recorder = LRRecorder(per_epoch=False)
    net = NeuralNet(
        module,
        torch.nn.MSELoss,
        callbacks=[recorder],
        max_epochs=5,
        optimizer__lr=0.1,
        batch_size=100)
    net.fit(X, y)

    lrs_per_epoch = net.history[:, 'batches', 'default_lr']
    for lr in lrs_per_epoch:
        assert lr == [0.1] * 10


def test_lr_recorder_custom_lr_per_epoch(data):
    module = MyModule()
    X, y = data

    param_group = [('layer1.*', {'lr': 0.01})]
    params_names = [name for name, _ in param_group]
    recorder = LRRecorder(group_names=params_names, per_epoch=True)

    net = NeuralNet(
        module,
        torch.nn.MSELoss,
        callbacks=[recorder],
        max_epochs=5,
        optimizer__lr=0.1,
        batch_size=100,
        optimizer__param_groups=param_group)
    net.fit(X, y)

    lrs = net.history[:, 'layer1.*']
    assert lrs == [0.01] * 5

    lrs = net.history[:, 'default_lr']
    assert lrs == [0.10] * 5


def test_lr_recorder_custom_lr_per_batch(data):
    module = MyModule()
    X, y = data

    param_group = [('layer1.*', {'lr': 0.01})]
    params_names = [name for name, _ in param_group]
    recorder = LRRecorder(group_names=params_names, per_epoch=False)

    net = NeuralNet(
        module,
        torch.nn.MSELoss,
        callbacks=[recorder],
        max_epochs=5,
        optimizer__lr=0.1,
        batch_size=100,
        optimizer__param_groups=param_group)
    net.fit(X, y)

    lrs_per_epoch_layer1 = net.history[:, 'batches', 'layer1.*']
    for lr in lrs_per_epoch_layer1:
        assert lr == [0.01] * 10

    lrs_per_epoch_default = net.history[:, 'batches', 'default_lr']
    for lr in lrs_per_epoch_default:
        assert lr == [0.1] * 10


# TODO: Test HistorySaver#
# TODO: Test TensorboardXLogger
