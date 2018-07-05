"""Pytorch utilities"""
import numpy as np
import torch


def set_requires_grad(module, name, value):
    """Set requires_grad to ``value`` in ``module`` with parameter
    names that start with ``name``

    Parameters
    ----------
    module: :class:`torch.nn.Module`
        Module to modify

    name: str
        Parameter name to query

    value: bool
        Value of requires_grad to set

    """
    filter_params = (p for n, p in module.named_parameters()
                     if n.startswith(name))
    for p in filter_params:
        p.requires_grad_(value)


def simulate_lrs(lr_sch_class, steps, initial_lr, **kwargs):
    """
    Simulates the learning rate scheduler.

    Parameters
    ----------
    lr_sch_class: class
        Class of pytorch learning rate scheduler

    steps: int
        Number of steps to simulate

    initial_lr: float
        Initial learning rate

    Returns
    -------
    numpy array

    """
    test = torch.ones(1, requires_grad=True)
    opt = torch.optim.SGD([{'params': test, 'lr': initial_lr}])
    sch = lr_sch_class(opt, **kwargs)

    has_batch_step = (hasattr(lr_sch_class, 'batch_step')
                      and callable(lr_sch_class.batch_step))
    lrs = []
    for _ in range(steps):
        sch.batch_step() if has_batch_step else sch.step()
        lrs.append(sch.get_lr()[0])

    return np.array(lrs)
