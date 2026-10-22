"""Pick the final training config from sweep results.

Chooses the best sweep run, preferring the medium ResNet-18-style model when
it is within 0.5% of the top score (short sweeps favor small models; the
larger model is the safer long-schedule choice). Applies the final training
settings (224px, 90 epochs, EMA) and writes auto_final.json.
"""
import glob
import json
import os

SWEEP_GLOB = "experiments/sweep_*/fold0_results.json"
OUT = "experiments/configs/auto_final.json"

FINAL_OVERRIDES = {
    "run_name": "cv_final",
    "image_size": 224,
    "num_epochs": 90,
    "warmup_epochs": 5,
    "batch_size": 128,
    "num_workers": 12,
    "num_folds": 5,
    "val_fold": 0,
    "ema": True,
    "ema_decay": 0.999,
    "eval_ema": True,
}


def main():
    results = []
    for path in sorted(glob.glob(SWEEP_GLOB)):
        with open(path) as f:
            results.append(json.load(f))
    if not results:
        print("No sweep results found; leaving final.json untouched")
        return 1

    results.sort(key=lambda r: r["best_val_acc"], reverse=True)
    best = results[0]
    for r in results:
        if 8e6 <= r.get("params", 0) <= 30e6 and \
                best["best_val_acc"] - r["best_val_acc"] <= 0.005:
            best = r
            break

    cfg = dict(best["config"])
    cfg.update(FINAL_OVERRIDES)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Selected {best['run_name']} "
          f"(val_acc={best['best_val_acc']:.4f}, "
          f"params={best.get('params')}) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
