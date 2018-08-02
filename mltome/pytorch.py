"""Pytorch utilities"""


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
