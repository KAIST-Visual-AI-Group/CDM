"""Utility functions for DNA discrete diffusion SMC.

Adapted from texts_mdm/utils.py.
"""

import copy
import logging

import lightning
import numpy as np
import torch

try:
    from timm.scheduler import CosineLRScheduler
except ImportError:
    CosineLRScheduler = None


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for lvl in ('debug', 'info', 'warning', 'error', 'exception', 'fatal', 'critical'):
        setattr(
            logger, lvl,
            lightning.pytorch.utilities.rank_zero_only(getattr(logger, lvl)),
        )
    return logger


class CosineDecayWarmupLRScheduler(
    *([CosineLRScheduler] if CosineLRScheduler is not None else []),
    torch.optim.lr_scheduler._LRScheduler,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_epoch = -1
        self.step(epoch=0)

    def step(self, epoch=None):
        if epoch is None:
            self._last_epoch += 1
        else:
            self._last_epoch = epoch
        if self.t_in_epochs:
            super().step(epoch=self._last_epoch)
        else:
            super().step_update(num_updates=self._last_epoch)


# ══════════════════════════════════════════════════════════════
#  Training utilities
# ══════════════════════════════════════════════════════════════


class EMA:
    """Exponential Moving Average of model parameters (for twist net)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model, decay=None):
        decay = self.decay if decay is None else decay
        for p_ema, p in zip(self.shadow.parameters(), model.parameters()):
            if p.requires_grad:
                p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def make_linear_decay_scheduler(optimizer, total_steps, decay_start_frac):
    decay_start = int(total_steps * decay_start_frac)

    def _lr_lambda(step):
        if step <= decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / (total_steps - decay_start))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    return scheduler, decay_start


def ess_normalized(log_w, dim):
    n = log_w.shape[dim]
    w = torch.softmax(log_w, dim=dim)
    ess = 1.0 / w.pow(2).sum(dim=dim).clamp(min=1e-12)
    return (ess / n).mean().item()


def ess_summary(ess_steps):
    if not ess_steps:
        return float("nan"), float("nan")
    return float(np.mean(ess_steps)), float(np.min(ess_steps))
