import torch.nn as nn

from mltome.pytorch import set_requires_grad


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
