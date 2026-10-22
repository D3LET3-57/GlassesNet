"""Create the configs for the fast CPU hyperparameter screening sweep."""
import json
import os

BASE = "experiments/configs/final.json"
OUT = "experiments/configs/hp_gpu"
os.makedirs(OUT, exist_ok=True)

base = json.load(open(BASE))
common = dict(base)
common.update({
    "output_dir": "experiments/hp_gpu",
    "image_size": 128,
    "num_epochs": 20,
    "warmup_epochs": 2,
    "limit_per_class": 0,
    "num_workers": 8,
    "device": "cuda",
    "ema": False,
    "eval_ema": False,
    "amp": True,
    "seed": 42,
    "split_seed": 42,
    "batch_size": 128,
})


def write(name, **overrides):
    cfg = dict(common)
    cfg.update(overrides)
    cfg["run_name"] = name
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", name)


for lr in (1e-4, 3e-4, 1e-3, 1e-2):
    write(f"lr_{lr:g}", lr=lr)

for act in ("relu", "gelu", "silu"):
    write(f"act_{act}", activation=act)

for bs in (32, 64, 128, 256):
    write(f"bs_{bs}", batch_size=bs)
