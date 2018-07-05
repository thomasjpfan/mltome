import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
from skorch.callbacks.lr_scheduler import CyclicLR

from mltome.pytorch import set_requires_grad
from mltome.pytorch import simulate_lrs


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)


def test_set_requires_grad():
    module = MyModule()
    set_requires_grad(module, 'layer1', False)

    for p in module.layer1.parameters():
        assert not p.requires_grad

    for p in module.layer2.parameters():
        assert p.requires_grad


def test_simulate_lrs_epoch_step():
    lrs = simulate_lrs(StepLR, 6, 1, step_size=2)
    expected = np.array([1.0, 1.0, 0.1, 0.1, 0.01, 0.01])
    assert np.allclose(expected, lrs)


def test_simulate_lrs_batch_step():
    lrs = simulate_lrs(CyclicLR, 11, 1, base_lr=1, max_lr=5, step_size=4)
    expected = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3])
    assert np.allclose(expected, lrs)
