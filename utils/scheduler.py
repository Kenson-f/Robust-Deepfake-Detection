"""
Learning Rate Schedulers
"""
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    LinearLR,
    SequentialLR,
    CosineAnnealingWarmRestarts
)


def get_scheduler(optimizer, cfg, steps_per_epoch: int):
    """
    Returns scheduler based on config.
    Supports: cosine, onecycle, warmup_cosine, cosine_restarts
    """
    sched_name = getattr(cfg, 'scheduler', 'warmup_cosine')

    if sched_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-7)

    elif sched_name == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=[cfg.lr_backbone, cfg.lr_head],
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.epochs,
            pct_start=0.1,
            anneal_strategy='cos'
        )

    elif sched_name == 'warmup_cosine':
        warmup_epochs = max(1, int(cfg.epochs * 0.1))
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cfg.epochs - warmup_epochs, eta_min=1e-7)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    elif sched_name == 'cosine_restarts':
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    return None
