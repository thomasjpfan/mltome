"""Skorch callbacks"""
import os
from itertools import product
from collections import defaultdict
from contextlib import suppress

from tensorboardX import SummaryWriter

from skorch.callbacks.base import Callback


class MetricsLogger(Callback):
    """Base class for creating metrics loggers. Child classes
    must implement ``update_batch_values`` and ``update_epoch_values``
    to use this logger

    Parameters
    ----------

    batch_targets : list of str
        Name of metrics to log at the end of each batch

    epoch_targets : list of str
        Name of metrics to log at the end of each epoch

    """

    def __init__(self, batch_targets=None, epoch_targets=None):

        self.batch_targets = batch_targets
        self.epoch_targets = epoch_targets

    def update_batch_values(self, values, idx):
        raise NotImplementedError("Not implemented")

    def update_epoch_values(self, values, idx):
        raise NotImplementedError("Not implemented")

    def on_batch_end(self, net, **kwargs):
        if self.batch_targets is None:
            return
        batch_idx = self._get_batch_idx(net)

        values = {}
        for name in self.batch_targets:
            values[name] = net.history[-1, 'batches', name, -1]

        self.update_batch_values(values, batch_idx)

    def on_epoch_end(self, net, **kwargs):
        if self.epoch_targets is None:
            return

        epoch = len(net.history)

        values = {}
        for name in self.epoch_targets:
            values[name] = net.history[-1, name]

        self.update_epoch_values(values, epoch)

    def _get_batch_idx(self, net):
        if not net.history:
            return -1
        epoch = len(net.history) - 1
        current_batch_idx = len(net.history[-1, 'batches']) - 1
        batch_cnt = len(net.history[-2, 'batches']) if epoch >= 1 else 0
        return epoch * batch_cnt + current_batch_idx


class LRRecorder(Callback):
    """Records learning rate from optimizer

    Parameters
    ----------

    group_names : list of str
        Name of each parameter group

    per_epoch : bool
        If true, record learning rate for each epoch, otherwise
        record each batch

    default_group : str
        Name of default learning rate group

    """

    def __init__(self,
                 group_names=None,
                 per_epoch=True,
                 default_group="default_lr"):
        self.group_names = group_names
        self.per_epoch = per_epoch
        self.default_group = default_group

    def initialize(self):
        group_names = self.group_names
        if group_names is None:
            group_names = []
        if self.default_group:
            group_names.append(self.default_group)
        self.group_names_ = group_names

    def on_train_begin(self, net, **kwargs):
        self.optimizer_ = net.optimizer_

    def on_epoch_end(self, net, **kwargs):
        if not self.per_epoch:
            return
        history = net.history
        pgroups = self.optimizer_.param_groups

        for pgroup, name in zip(pgroups, self.group_names_):
            history.record(name, pgroup['lr'])

    def on_batch_end(self, net, **kwargs):
        if self.per_epoch:
            return
        history = net.history
        pgroups = self.optimizer_.param_groups

        for pgroup, name in zip(pgroups, self.group_names_):
            history.record_batch(name, pgroup['lr'])


class HistorySaver(Callback):
    """Save history per callbacks

    Parameters
    ----------

    target : str
        Filename to store history

    """

    def __init__(self, target):
        self.target = target

    def on_epoch_end(self, net, **kwargs):
        net.save_history(self.target)


class TensorboardXLogger(MetricsLogger):
    """TensorboardX callbacks

    Parameters
    ----------

    name : str
        Name of current run

    batch_targets : list of str
        Targets to record per batch

    epoch_targets : list of str
        Targets to record per epoch

    batch_groups : list of str
        Groups to record per batch

    epoch_groupss : list of str
        Groups to record per epoch

    """

    def __init__(self,
                 name,
                 batch_targets=None,
                 epoch_targets=None,
                 batch_groups=None,
                 epoch_groups=None,
                 root_dir='artifacts/runs'):
        self.log_dir = os.path.join(root_dir, name)

        self.batch_groups = batch_groups or []
        self.epoch_groups = epoch_groups or []
        self.batch_targets = batch_targets or []
        self.epoch_targets = epoch_targets or []

        super().__init__(
            batch_targets=batch_targets, epoch_targets=epoch_targets)

    def initialize(self):
        self.writer_ = SummaryWriter(log_dir=self.log_dir)
        self.batch_target_to_name_ = {}
        for g, t in product(self.batch_groups, self.batch_targets):
            if t.endswith(g):
                self.batch_target_to_name[t] = 'batch_' + g

        self.epoch_target_to_name_ = {}
        for g, t in product(self.epoch_groups, self.epoch_targets):
            if t.endswith(g):
                self.epoch_target_to_name[t] = 'epoch_' + g

    def update_batch_values(self, values, idx):
        vgroups = defaultdict(dict)
        for name, value in values.items():
            self.writer_.add_scalar(f'batch/{name}', value, idx)
            with suppress(KeyError):
                group = self.batch_target_to_name[name]
                vgroups[group][name] = value

        for group, values in vgroups.items():
            self.writer_.add_scalars(group, values, idx)

    def update_epoch_values(self, values, idx):
        vgroups = defaultdict(dict)
        for name, value in values.items():
            self.writer_.add_scalar(f'epoch/{name}', value, idx)
            with suppress(KeyError):
                group = self.epoch_target_to_name[name]
                vgroups[group][name] = value

        for group, values in vgroups.items():
            self.writer_.add_scalars(group, values, idx)

    def on_train_end(self, net, **kwargs):
        self.writer_.close()
