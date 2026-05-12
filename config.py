"""
Training Configurations for NTIRE 2026 Robust Deepfake Detection v2
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────────────
    image_size: int = 224
    num_workers: int = 4

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone: str = 'efficientnet_b4'
    pretrained: bool = True
    dropout_rate: float = 0.3

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 20            # 1000 images trains fast; 20 is plenty with early stop
    batch_size: int = 32
    accum_steps: int = 2        # gradient accumulation → effective batch = 64
    seed: int = 42

    # ── MixUp ─────────────────────────────────────────────────────────────────
    use_mixup: bool = True
    mixup_alpha: float = 0.2

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr_backbone: float = 1e-5   # smaller → more stable fine-tuning
    lr_head: float = 1e-4
    weight_decay: float = 2e-4

    # ── Loss ──────────────────────────────────────────────────────────────────
    bce_weight: float = 0.4
    focal_weight: float = 0.6   # more focal → focus on hard examples
    label_smoothing: float = 0.05

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler: str = 'warmup_cosine'

    # ── Checkpointing ─────────────────────────────────────────────────────────
    save_every: int = 5
    patience: int = 7           # stop early if no improvement for 7 epochs

    def __repr__(self):
        return (f"Config(backbone={self.backbone}, epochs={self.epochs}, "
                f"bs={self.batch_size}×{self.accum_steps}, img={self.image_size})")


@dataclass
class FastConfig(Config):
    """Quick sanity-check run."""
    backbone: str = 'efficientnet_b3'
    epochs: int = 10
    batch_size: int = 32
    accum_steps: int = 1
    patience: int = 5
    use_mixup: bool = False


@dataclass
class XceptionConfig(Config):
    """
    EfficientNet-B4 at 299px — Xception-like resolution.
    Good balance of speed and accuracy.
    """
    backbone: str = 'efficientnet_b4'
    image_size: int = 299
    epochs: int = 25
    batch_size: int = 16
    accum_steps: int = 4        # effective batch = 64
    lr_backbone: float = 8e-6
    lr_head: float = 8e-5
    dropout_rate: float = 0.35
    use_mixup: bool = True
    mixup_alpha: float = 0.3


@dataclass
class StrongConfig(Config):
    """
    Best competition config. ConvNeXt-Base at 256px.
    Recommended if you have 8GB+ VRAM.
    """
    backbone: str = 'convnext_base'
    image_size: int = 256
    epochs: int = 25
    batch_size: int = 16
    accum_steps: int = 4
    lr_backbone: float = 5e-6
    lr_head: float = 5e-5
    weight_decay: float = 5e-4
    dropout_rate: float = 0.4
    label_smoothing: float = 0.08
    scheduler: str = 'cosine_restarts'
    patience: int = 15
    use_mixup: bool = True
    mixup_alpha: float = 0.3


@dataclass
class ViTConfig(Config):
    """Vision Transformer — captures global face structure."""
    backbone: str = 'vit_small_patch16_224'
    epochs: int = 20
    batch_size: int = 32
    accum_steps: int = 2
    lr_backbone: float = 3e-6
    lr_head: float = 5e-5
    weight_decay: float = 5e-4
    dropout_rate: float = 0.35
    scheduler: str = 'warmup_cosine'
    patience: int = 12


CONFIGS = {
    'default':  Config,
    'fast':     FastConfig,
    'xception': XceptionConfig,
    'strong':   StrongConfig,
    'vit':      ViTConfig,
}


def get_config(name: str = 'default') -> Config:
    cls = CONFIGS.get(name, Config)
    return cls()
