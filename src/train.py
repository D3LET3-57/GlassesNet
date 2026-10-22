"""
Training loop for the from-scratch CNN.

Features: AMP, AdamW/SGD-family, warmup + cosine schedule, label smoothing,
gradient clipping, EMA weights, resumable checkpoints and per-epoch logs of
loss / accuracy / F1 (the assignment's required metrics).
"""
import copy
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

from src.config import Config
from src.cnn_model import build_model
from src.dataset import build_loaders


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_criterion(cfg: Config) -> nn.Module:
    if cfg.loss == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    return nn.MSELoss()


def build_optimizer(cfg: Config, model: nn.Module):
    if cfg.opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    if cfg.opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
    if cfg.opt == "gd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    if cfg.opt in ("momentum", "nag"):
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               momentum=cfg.momentum,
                               nesterov=(cfg.opt == "nag"),
                               weight_decay=cfg.weight_decay)
    if cfg.opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr,
                               momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.opt}")


def build_scheduler(cfg: Config, optimizer):
    if not cfg.anneal:
        return None
    warmup = max(0, cfg.warmup_epochs)
    total = max(1, cfg.num_epochs)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        progress = (epoch - warmup) / max(1, total - warmup)
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """Exponential moving average of weights and buffers."""

    def __init__(self, model: nn.Module, decay: float):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, p in zip(self.ema.parameters(), model.parameters()):
            e.mul_(d).add_(p.detach(), alpha=1 - d)
        for e, b in zip(self.ema.buffers(), model.buffers()):
            e.copy_(b)


def _metrics(logits: torch.Tensor, labels: torch.Tensor):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (
        precision + recall) else 0.0
    return correct / total, f1


def _to_targets(labels: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        t = torch.zeros(labels.size(0), 2, device=labels.device)
        t.scatter_(1, labels.unsqueeze(1), 1.0)
        return t
    return labels


def train_epoch(model, loader, criterion, optimizer, scaler, device, cfg):
    model.train()
    use_amp = cfg.amp and device.type == "cuda"
    running_loss = 0.0
    logits_all, labels_all = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = _to_targets(labels, cfg.loss)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        logits_all.append(logits.detach())
        labels_all.append(labels)

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)
    acc, f1 = _metrics(logits_all, labels_all)
    return running_loss / len(loader.dataset), acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg):
    model.eval()
    use_amp = cfg.amp and device.type == "cuda"
    running_loss = 0.0
    logits_all, labels_all = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = _to_targets(labels, cfg.loss)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)
        logits_all.append(logits.float())
        labels_all.append(labels)

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)
    acc, f1 = _metrics(logits_all, labels_all)
    return running_loss / len(loader.dataset), acc, f1


def _state_to_cpu(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(cfg: Config, resume: bool = False):
    """Train one model. Returns (model, history)."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    tag = f"fold{cfg.val_fold}"

    train_loader, val_loader, _, _, train_idx, val_idx = build_loaders(cfg)
    cfg_snapshot = cfg.to_dict()
    cfg_snapshot["train_size"] = len(train_idx)
    cfg_snapshot["val_size"] = len(val_idx)

    model = build_model(cfg).to(device)
    if cfg.init_from:
        ckpt = torch.load(cfg.init_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Initialized weights from {cfg.init_from} "
              f"(epoch {ckpt.get('epoch')}, "
              f"val_acc={ckpt.get('val_acc', float('nan')):.4f})")
    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema else None

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "ema_val_loss": [], "ema_val_acc": [], "ema_val_f1": [],
        "lr": [], "epoch_time": [],
    }
    start_epoch = 0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = -1
    best_used_ema = False

    last_path = os.path.join(run_dir, f"{tag}_last.pt")
    best_path = os.path.join(run_dir, f"{tag}_best.pt")
    metrics_path = os.path.join(run_dir, f"{tag}_metrics.jsonl")

    if resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        if ema is not None and ckpt.get("ema_state_dict"):
            ema.ema.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_epoch = ckpt.get("best_epoch", -1)
        best_used_ema = ckpt.get("best_used_ema", False)
        hist_path = os.path.join(run_dir, f"{tag}_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
        print(f"Resumed {run_dir} at epoch {start_epoch} "
              f"(best val acc {best_val_acc:.4f})")

    print(f"Device: {device} | params: {model.num_parameters():,} | "
          f"train {len(train_idx)} | val {len(val_idx)} | {cfg.run_name}")
    print(f"Config: image_size={cfg.image_size} opt={cfg.opt} lr={cfg.lr} "
          f"loss={cfg.loss} ls={cfg.label_smoothing} bs={cfg.batch_size} "
          f"epochs={cfg.num_epochs} anneal={cfg.anneal} ema={cfg.ema}")

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg)
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device, cfg)

        ema_loss = ema_acc = ema_f1 = float("nan")
        if ema is not None:
            ema.update(model)
            ema_loss, ema_acc, ema_f1 = evaluate(
                ema.ema, val_loader, criterion, device, cfg)

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["ema_val_loss"].append(ema_loss)
        history["ema_val_acc"].append(ema_acc)
        history["ema_val_f1"].append(ema_f1)
        history["lr"].append(lr)
        history["epoch_time"].append(dt)

        score = ema_acc if (ema is not None and cfg.eval_ema) else val_acc
        if score > best_val_acc:
            best_val_acc = score
            best_val_f1 = ema_f1 if (ema is not None and cfg.eval_ema) else val_f1
            best_epoch = epoch + 1
            best_used_ema = ema is not None and cfg.eval_ema
            best_model = ema.ema if best_used_ema else model
            torch.save({
                "model_state_dict": _state_to_cpu(best_model),
                "config": cfg_snapshot,
                "epoch": epoch + 1,
                "val_acc": score,
                "val_f1": best_val_f1,
                "used_ema": best_used_ema,
            }, best_path)

        if (epoch + 1) % cfg.log_interval == 0:
            print(f"Epoch {epoch + 1:3d}/{cfg.num_epochs} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | "
                  f"ema acc {ema_acc:.4f} | lr {lr:.2e} | {dt:.1f}s",
                  flush=True)

        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                "ema_val_loss": ema_loss, "ema_val_acc": ema_acc,
                "ema_val_f1": ema_f1, "lr": lr, "epoch_time": dt,
            }) + "\n")

        torch.save({
            "model_state_dict": _state_to_cpu(model),
            "ema_state_dict": _state_to_cpu(ema.ema) if ema else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict(),
            "config": cfg_snapshot,
            "epoch": epoch + 1,
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "best_epoch": best_epoch,
            "best_used_ema": best_used_ema,
        }, last_path)

        with open(os.path.join(run_dir, f"{tag}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    results = {
        "run_name": cfg.run_name,
        "fold": cfg.val_fold,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "final_val_acc": history["val_acc"][-1],
        "final_ema_val_acc": (history["ema_val_acc"][-1]
                              if history["ema_val_acc"] else None),
        "best_epoch": best_epoch,
        "used_ema": best_used_ema,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "params": model.num_parameters(),
        "config": cfg_snapshot,
    }
    with open(os.path.join(run_dir, f"{tag}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Best val acc {best_val_acc:.4f} at epoch {best_epoch} "
          f"({'EMA' if best_used_ema else 'raw'})", flush=True)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, history
