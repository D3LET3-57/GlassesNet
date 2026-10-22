"""Generate the hyperparameter figures for the verbose report.

Uses only matplotlib and the JSON experiment artifacts. NumPy is not imported.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "report/figures_hp"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


def load(path):
    with open(path) as f:
        return json.load(f)


def acc_pct(result):
    return 100.0 * result["best_val_acc"]


def annotate_bars(ax, bars, values, fmt="{:.2f}%"):
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, fmt.format(v),
                ha="center", va="bottom", fontsize=9)


def bar_plot(labels, values, title, ylabel, path, color="#3b6ea5",
             ylim=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=color)
    annotate_bars(ax, bars, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        lo, hi = min(values), max(values)
        pad = max(0.05, 0.15 * (hi - lo + 1e-9))
        ax.set_ylim(lo - pad, hi + pad)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", path)


# ---------------------------------------------------------------------------
# New controlled screening: 128px, 20 epochs, full data, fold 0.
# ---------------------------------------------------------------------------
def hp_result(pattern):
    hits = glob.glob(pattern)
    return load(sorted(hits)[0]) if hits else None


lr_rows = []
for p in sorted(glob.glob("experiments/hp_gpu/lr_*/fold0_results.json")):
    r = load(p)
    lr_rows.append((float(r["config"]["lr"]), acc_pct(r), r["run_name"]))
lr_rows.sort()

act_rows = []
for p in sorted(glob.glob("experiments/hp_gpu/act_*/fold0_results.json")):
    r = load(p)
    act_rows.append((r["config"]["activation"], acc_pct(r), r["run_name"]))
act_order = {"relu": 0, "gelu": 1, "silu": 2, "leaky_relu": 3}
act_rows.sort(key=lambda x: act_order.get(x[0], 9))

bs_rows = []
for p in sorted(glob.glob("experiments/hp_gpu/bs_*/fold0_results.json")):
    r = load(p)
    bs_rows.append((int(r["config"]["batch_size"]), acc_pct(r), r["run_name"]))
bs_rows.sort()

if lr_rows:
    xs = [x for x, _, _ in lr_rows]
    ys = [y for _, y, _ in lr_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogx(xs, ys, "o-", color="#3b6ea5", lw=2, ms=7)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Learning rate (AdamW, log scale)")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("Learning-rate screening (128px, 20 epochs, fold 0)")
    ax.grid(alpha=0.3)
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo - max(0.1, 0.3 * (hi - lo)), hi + max(0.1, 0.3 * (hi - lo)))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lr.png"), dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", os.path.join(OUT, "lr.png"))

    # Loss curves for the learning-rate runs.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for x, _, name in lr_rows:
        h = load(f"experiments/hp_gpu/{name}/fold0_history.json")
        ep = range(1, len(h["val_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], lw=1.7, label=f"lr={x:g}")
        axes[1].plot(ep, h["val_loss"], lw=1.7, label=f"lr={x:g}")
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Learning-rate effect on optimization", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "lr_curves.png"), dpi=170,
                bbox_inches="tight")
    plt.close()
    print("saved", os.path.join(OUT, "lr_curves.png"))

if act_rows:
    labels = [a for a, _, _ in act_rows]
    vals = [v for _, v, _ in act_rows]
    bar_plot(labels, vals, "Activation screening (128px, 20 epochs, fold 0)",
             "Best validation accuracy (%)",
             os.path.join(OUT, "activation.png"))

if bs_rows:
    xs = [x for x, _, _ in bs_rows]
    ys = [y for _, y, _ in bs_rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(xs, ys, "o-", color="#3b6ea5", lw=2, ms=7)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("Batch size (log scale)")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("Batch-size screening (128px, 20 epochs, fold 0)")
    ax.grid(alpha=0.3)
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo - max(0.1, 0.3 * (hi - lo)), hi + max(0.1, 0.3 * (hi - lo)))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "batch.png"), dpi=170, bbox_inches="tight")
    plt.close()
    print("saved", os.path.join(OUT, "batch.png"))

# ---------------------------------------------------------------------------
# Existing full 160px/30-epoch sweep: hidden size and optimizer.
# ---------------------------------------------------------------------------
hidden_specs = [
    ("sweep_03_small_adamw", "Small\n32-64-128-256\n2.71M"),
    ("sweep_01_r18_adamw", "Medium\n64-128-256-512\n11.18M"),
    ("sweep_04_r34_adamw", "Deep\n64-128-256-512\n21.29M"),
]
hidden_labels, hidden_vals = [], []
for run, label in hidden_specs:
    r = load(f"experiments/{run}/fold0_results.json")
    hidden_labels.append(label)
    hidden_vals.append(acc_pct(r))
bar_plot(hidden_labels, hidden_vals,
         "Model-size screening (160px, 30 epochs, fold 0)",
         "Best validation accuracy (%)",
         os.path.join(OUT, "hidden.png"))

opt_specs = [
    ("sweep_01_r18_adamw", "AdamW\nlr=1e-3"),
    ("sweep_02_r18_sgd", "SGD + momentum\nlr=0.1"),
]
opt_labels, opt_vals = [], []
for run, label in opt_specs:
    r = load(f"experiments/{run}/fold0_results.json")
    opt_labels.append(label)
    opt_vals.append(acc_pct(r))
bar_plot(opt_labels, opt_vals,
         "Optimizer screening (160px, 30 epochs, fold 0)",
         "Best validation accuracy (%)",
         os.path.join(OUT, "optimizer.png"))

print("\nLR:", lr_rows)
print("Activation:", act_rows)
print("Batch:", bs_rows)
print("Hidden:", list(zip(hidden_labels, hidden_vals)))
print("Optimizer:", list(zip(opt_labels, opt_vals)))
