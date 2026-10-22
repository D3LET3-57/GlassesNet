"""Generate report/notebook figures from saved experiment JSON files."""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    with open(path) as f:
        return json.load(f)


def _results(exp_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(exp_dir, "**",
                                              "*_results.json"),
                                  recursive=True)):
        r = _load(path)
        c = r.get("config", {})
        rows.append({
            "name": r.get("run_name", os.path.basename(os.path.dirname(path))),
            "fold": r.get("fold"),
            "val_acc": r.get("best_val_acc", 0.0),
            "params": r.get("params", 0),
            "size": c.get("image_size"),
            "epochs": c.get("num_epochs"),
            "opt": c.get("opt"),
            "lr": c.get("lr"),
            "path": path,
        })
    return rows


def plot_sweep(exp_dir, out_path, title="Sweep: validation accuracy"):
    rows = _results(exp_dir)
    rows = [r for r in rows if "sweep_" in r["name"]]
    if not rows:
        print(f"No sweep results under {exp_dir}")
        return
    rows.sort(key=lambda r: r["val_acc"])
    names = [f"{r['name']}\n({r['params'] / 1e6:.1f}M, {r['opt']}, "
             f"{r['size']}px)" for r in rows]
    vals = [100 * r["val_acc"] for r in rows]

    plt.figure(figsize=(10, max(4, 0.75 * len(rows))))
    bars = plt.barh(names, vals, color="#3b6ea5")
    for b, v in zip(bars, vals):
        plt.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.2f}%",
                 va="center", fontsize=9)
    plt.xlim(min(vals) - 2 if min(vals) > 2 else 0, 100.5)
    plt.xlabel("Best validation accuracy (%)")
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_history(run_dir, tag, out_path, title="Training history"):
    hist_path = os.path.join(run_dir, f"{tag}_history.json")
    if not os.path.exists(hist_path):
        print(f"Missing {hist_path}")
        return
    h = _load(hist_path)
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(epochs, h["train_loss"], label="Train", lw=2)
    axes[0].plot(epochs, h["val_loss"], label="Val", lw=2)
    if any(v == v for v in h.get("ema_val_loss", [])):
        axes[0].plot(epochs, h["ema_val_loss"], label="Val (EMA)", lw=2, ls="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, h["train_acc"], label="Train", lw=2)
    axes[1].plot(epochs, h["val_acc"], label="Val", lw=2)
    if any(v == v for v in h.get("ema_val_acc", [])):
        axes[1].plot(epochs, h["ema_val_acc"], label="Val (EMA)", lw=2, ls="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, h["train_f1"], label="Train", lw=2)
    axes[2].plot(epochs, h["val_f1"], label="Val", lw=2)
    if any(v == v for v in h.get("ema_val_f1", [])):
        axes[2].plot(epochs, h["ema_val_f1"], label="Val (EMA)", lw=2, ls="--")
    axes[2].set_title("F1 score")
    axes[2].set_xlabel("Epoch")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_cv(exp_dir, out_curves, out_bar,
            title="5-fold cross-validation"):
    hist_paths = sorted(glob.glob(os.path.join(exp_dir,
                                               "fold*_history.json")))
    if not hist_paths:
        print(f"No CV histories under {exp_dir}")
        return

    plt.figure(figsize=(9, 5))
    best = []
    for path in hist_paths:
        h = _load(path)
        tag = os.path.basename(path).replace("_history.json", "")
        acc = h.get("ema_val_acc")
        if not any(v == v for v in acc):
            acc = h["val_acc"]
        plt.plot(range(1, len(acc) + 1), acc, lw=1.8, label=tag)
        best.append(max(acc))
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_curves), exist_ok=True)
    plt.savefig(out_curves, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_curves}")

    folds = [os.path.basename(p).replace("_history.json", "")
             for p in hist_paths]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(folds, [100 * b for b in best], color="#3b6ea5")
    mean = 100 * sum(best) / len(best)
    plt.axhline(mean, color="crimson", ls="--",
                label=f"mean {mean:.2f}%")
    for b, v in zip(bars, best):
        plt.text(b.get_x() + b.get_width() / 2, 100 * v + 0.02,
                 f"{100 * v:.2f}%", ha="center", fontsize=9)
    plt.ylim(max(0, min(100 * v for v in best) - 1.5), 100.2)
    plt.ylabel("Best validation accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_bar), exist_ok=True)
    plt.savefig(out_bar, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_bar}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="report/figures")
    p.add_argument("--sweep-dir", default="experiments")
    p.add_argument("--cv-dir", default="experiments/cv_final")
    p.add_argument("--sweep-best-run", default="sweep_01_r18_adamw")
    args = p.parse_args()

    plot_sweep(args.sweep_dir,
               os.path.join(args.out_dir, "sweep_comparison.png"))
    plot_history(os.path.join(args.sweep_dir, args.sweep_best_run), "fold0",
                 os.path.join(args.out_dir, "sweep_best_history.png"),
                 title=f"Sweep best: {args.sweep_best_run}")
    plot_cv(args.cv_dir,
            os.path.join(args.out_dir, "cv_curves.png"),
            os.path.join(args.out_dir, "cv_scores.png"))


if __name__ == "__main__":
    main()
