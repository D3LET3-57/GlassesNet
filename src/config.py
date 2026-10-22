"""
Configuration for the from-scratch glasses classifier.

Every assignment-required hyperparameter (lr, momentum, num_hidden, sizes,
activation, loss, opt, batch_size, anneal) is exposed here, together with the
extra knobs used by the convolutional training recipe.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import json


@dataclass
class Config:
    """Hyperparameter configuration for the from-scratch CNN."""

    # ----------------------------- data ---------------------------------
    train_csv: str = "data/kaggle/kaggle/train.csv"
    train_dir: str = "data/kaggle/kaggle"
    test_csv: str = "data/Dataset/test_dataset.csv"
    test_dir: str = "data/Dataset/testimages"
    output_dir: str = "experiments"

    image_size: int = 224
    num_folds: int = 5
    val_fold: int = 0
    seed: int = 42                  # model init / training seed
    split_seed: int = 42            # fixed fold definition seed
    num_workers: int = 8
    limit_per_class: int = 0        # >0 only for smoke tests
    test_limit: int = 0             # >0 only for smoke tests

    # ------------------------- architecture -----------------------------
    num_hidden: int = 4             # number of conv stages
    sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    blocks_per_stage: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    activation: str = "relu"        # relu, leaky_relu, gelu, silu, sigmoid, tanh
    stem_width: int = 64
    dropout: float = 0.2

    # --------------------------- loss -----------------------------------
    loss: str = "cross_entropy"     # cross_entropy, mse
    label_smoothing: float = 0.1

    # ------------------------- optimizer --------------------------------
    opt: str = "adamw"              # gd, momentum, nag, sgd, adam, adamw
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    grad_clip: float = 1.0

    # -------------------------- schedule --------------------------------
    num_epochs: int = 100
    warmup_epochs: int = 5
    anneal: bool = True             # cosine decay
    min_lr_ratio: float = 0.01

    # -------------------------- training --------------------------------
    batch_size: int = 128
    amp: bool = True
    ema: bool = True
    ema_decay: float = 0.999
    eval_ema: bool = True
    log_interval: int = 1

    # ------------------------ augmentation ------------------------------
    aug_random_resized_crop: bool = True
    aug_scale_min: float = 0.65
    aug_hflip: bool = True
    aug_rotation: float = 7.0
    aug_color_jitter: float = 0.2
    aug_random_erasing: float = 0.25

    # -------------------------- artifacts -------------------------------
    run_name: str = "run"
    device: str = "cuda"
    init_from: str = ""             # optional checkpoint to initialize weights

    # ------------------------- inference --------------------------------
    tta_views: int = 4
    predict_batch_size: int = 192

    def __post_init__(self):
        assert self.activation in (
            "relu", "leaky_relu", "gelu", "silu", "sigmoid", "tanh"
        ), f"Unknown activation: {self.activation}"
        assert self.loss in ("cross_entropy", "mse"), \
            f"Unknown loss: {self.loss}"
        assert self.opt in ("gd", "momentum", "nag", "sgd", "adam", "adamw"), \
            f"Unknown optimizer: {self.opt}"
        assert len(self.sizes) == self.num_hidden, \
            f"sizes ({len(self.sizes)}) must match num_hidden ({self.num_hidden})"
        assert len(self.blocks_per_stage) == self.num_hidden, \
            "blocks_per_stage must match num_hidden"
        assert self.num_folds >= 1

    @property
    def output_size(self) -> int:
        return 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
