"""
Batched test-time augmentation (TTA) and multi-model ensemble inference.

All network computation uses PyTorch. Probability artifacts are saved as a
PyTorch `.pt` file for analysis; pandas handles CSV I/O.
"""
import argparse
import glob
import json
import os

import pandas as pd
import torch
import torchvision.transforms as T

from src.config import Config
from src.cnn_model import build_model
from src.dataset import build_test_loader, MEAN, STD


def make_views(size: int, n_views: int):
    """Deterministic TTA views: identity, hflip, two zoom levels (x2)."""

    def view(resize_to: int, flip: bool) -> T.Compose:
        ops = [T.Resize(resize_to), T.CenterCrop(size)]
        if flip:
            ops.append(T.RandomHorizontalFlip(p=1.0))
        ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
        return T.Compose(ops)

    all_views = [
        view(size + 32, False),
        view(size + 32, True),
        view(int(round(size * 1.15)) + 32, False),
        view(int(round(size * 1.15)) + 32, True),
        view(size + 96, False),
        view(size + 96, True),
    ]
    return all_views[:n_views]


@torch.no_grad()
def predict_probs(model, loader, device, use_amp: bool) -> torch.Tensor:
    model.eval()
    chunks = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
        chunks.append(torch.softmax(logits.float(), dim=1))
    return torch.cat(chunks)


def load_checkpoint(path: str, device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg, ckpt


def run_ensemble(checkpoint_paths, output_csv, tta_views=4,
                 batch_size=192, num_workers=8, probs_out=None, limit=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    models, cfgs, ckpts = [], [], []
    for path in checkpoint_paths:
        model, cfg, ckpt = load_checkpoint(path, device)
        models.append(model)
        cfgs.append(cfg)
        ckpts.append(ckpt)
        print(f"Loaded {os.path.basename(path)} | "
              f"val_acc={ckpt.get('val_acc', float('nan')):.4f} "
              f"epoch={ckpt.get('epoch')} ema={ckpt.get('used_ema')}")

    cfg = cfgs[0]
    for c in cfgs:
        c.test_limit = limit
    sizes = sorted({int(c.image_size) for c in cfgs})
    print(f"Ensemble: {len(models)} model(s), {tta_views} TTA views, "
          f"image sizes {sizes}")

    per_model = []
    for mi, (model, model_cfg) in enumerate(zip(models, cfgs)):
        size = model_cfg.image_size
        views = make_views(size, tta_views)
        acc = None
        for vi, transform in enumerate(views):
            loader, _ = build_test_loader(model_cfg, transform, batch_size,
                                          num_workers)
            probs = predict_probs(model, loader, device, use_amp)
            acc = probs if acc is None else acc + probs
            print(f"  model {mi + 1}/{len(models)} ({size}px) view "
                  f"{vi + 1}/{len(views)}", flush=True)
        acc = acc / len(views)
        per_model.append(acc.cpu())
        torch.cuda.empty_cache()

    stacked = torch.stack(per_model)              # (M, N, 2)
    ensemble = stacked.mean(dim=0)                # (N, 2)
    preds_t = ensemble.argmax(dim=1)
    preds = preds_t.tolist()

    df = pd.read_csv(cfg.test_csv)
    if cfg.test_limit > 0:
        df = df.head(cfg.test_limit).reset_index(drop=True)
    submission = pd.DataFrame({
        "id": df["id"].astype(str),
        "glasses": preds,
    })
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    submission.to_csv(output_csv, index=False)

    votes = stacked.argmax(dim=2)                 # (M, N)
    unanimous = int((votes == preds_t[None, :]).all(dim=0).sum()) \
        if len(models) > 1 else len(preds)
    disagreement = 0
    if len(models) > 1:
        disagreement = int((votes.max(dim=0).values
                            != votes.min(dim=0).values).sum())

    summary = {
        "checkpoints": [os.path.basename(p) for p in checkpoint_paths],
        "tta_views": tta_views,
        "image_sizes": sizes,
        "n_test": len(preds),
        "glasses": int(preds_t.sum()),
        "no_glasses": int(len(preds) - int(preds_t.sum())),
        "mean_confidence": float(ensemble.max(dim=1).values.mean()),
        "unanimous": unanimous,
        "model_disagreements": disagreement,
        "val_accs": [float(c.get("val_acc", float("nan"))) for c in ckpts],
    }
    print(json.dumps(summary, indent=2))

    if probs_out:
        os.makedirs(os.path.dirname(probs_out) or ".", exist_ok=True)
        torch.save({
            "ids": df["id"].astype(str).tolist(),
            "ensemble_probs": ensemble.cpu(),
            "per_model_probs": stacked.cpu(),
            "preds": preds_t.cpu(),
        }, probs_out)
        with open(probs_out.replace(".pt", "_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Probabilities saved to {probs_out}")

    print(f"Submission saved to {output_csv} "
          f"({len(preds)} rows, {int(preds.sum())} glasses)")
    return submission, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="experiments/cv_final")
    p.add_argument("--pattern", default="fold*_best.pt")
    p.add_argument("--checkpoints", nargs="*", default=None)
    p.add_argument("--output", default="submission/submission.csv")
    p.add_argument("--tta-views", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--probs-out", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if args.checkpoints:
        paths = args.checkpoints
    else:
        paths = sorted(glob.glob(os.path.join(args.checkpoint_dir,
                                              args.pattern)))
    if not paths:
        raise SystemExit(f"No checkpoints matched {args.pattern} in "
                         f"{args.checkpoint_dir}")
    run_ensemble(paths, args.output, args.tta_views, args.batch_size,
                 args.num_workers, args.probs_out, args.limit)


if __name__ == "__main__":
    main()
