"""Build the final deliverable notebook from saved experiment artifacts.

Run after training:
    python scripts/build_notebook.py
    CUDA_VISIBLE_DEVICES="" jupyter nbconvert --to notebook --execute \
        --inplace notebooks/glasses_classifier.ipynb
"""
import os

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


cells = [
    md("""# DAC-301 Assignment: Glasses Detection from Scratch

Binary classification: does the person wear glasses?

**Rule compliance.** This notebook trains a convolutional neural network
**entirely from scratch** on the provided dataset. There are no pretrained
weights, no transfer learning, and no ViT/DDPM-type architectures. The
network is implemented with `torch.nn`. All model, training, metric and
inference computations use PyTorch; there are no NumPy imports in the
model/training/inference code. Data handling uses pandas/PIL and figures use
matplotlib.

**Pipeline.** From-scratch ResNet-style CNN -> hyperparameter sweep ->
5-fold cross-validation -> multi-view TTA + 5-model ensemble -> submission.
"""),
    md("## 1. Setup"),
    code("""import glob
import json
import os
import sys

import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import Image, display

sys.path.insert(0, os.path.abspath('..'))

from src.config import Config
from src.cnn_model import build_model
from src.dataset import (GlassesDataset, build_eval_transform,
                         build_train_transform, stratified_folds)

print(f"PyTorch {torch.__version__}")
print("Training ran on GPU via SLURM; this notebook is executed on CPU "
      "for inspection.")
"""),
    md("""## 2. Data

24,499 training images (512x512 RGB, balanced) and 5,412 test images.
Validation uses a deterministic **stratified 5-fold split**; each fold model
is trained on 80% and validated on the held-out 20%.
"""),
    code("""train_df = pd.read_csv('../data/kaggle/kaggle/train.csv')
test_df = pd.read_csv('../data/Dataset/test_dataset.csv')
print(f"Train: {len(train_df)} | Test: {len(test_df)}")
print(train_df['glasses'].value_counts().rename({0: 'no glasses', 1: 'glasses'}))

labels = torch.tensor(train_df['glasses'].tolist())
folds = stratified_folds(labels, num_folds=5, seed=42)
print("Fold sizes:", [(len(tr), len(va)) for tr, va in folds])
"""),
    code("""from PIL import Image as PILImage

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Training samples', fontsize=14)
for ax, (_, row) in zip(axes.flat, train_df.sample(10, random_state=7).iterrows()):
    img = PILImage.open(os.path.join('../data/kaggle/kaggle', row['file'])).convert('RGB')
    ax.imshow(img)
    ax.set_title('Glasses' if row['glasses'] == 1 else 'No glasses')
    ax.axis('off')
plt.tight_layout()
plt.savefig('../report/figures/sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
"""),
    md("""## 3. Model

A hand-written residual CNN (`src/cnn_model.py`): 7x7 stem, four residual
stages of BasicBlocks, global average pooling and a linear classifier. The
architecture is fully configurable through `Config` (`num_hidden`, `sizes`,
`blocks_per_stage`, `activation`, ...), satisfying the assignment's
configurable-hyperparameter requirement.
"""),
    code("""cfg_path = '../experiments/configs/auto_final.json'
if not os.path.exists(cfg_path):
    cfg_path = '../experiments/configs/final.json'
cfg = Config.from_json(cfg_path)
model = build_model(cfg)
print(model)
print(f"\\nTrainable parameters: {model.num_parameters():,}")
print("\\nConfigurable hyperparameters:")
for k, v in cfg.to_dict().items():
    print(f"  {k}: {v}")
"""),
    md("""### Training recipe

- Cross-entropy with label smoothing 0.1
- AdamW, warmup + cosine learning-rate decay
- Mixed precision (AMP), gradient clipping
- Augmentation: RandomResizedCrop, horizontal flip, rotation, color jitter,
  random erasing
- EMA weights, best checkpoint selected on validation accuracy
- Loss / accuracy / F1 logged every epoch

Reproduce a fold:

```bash
python -m src.run --config experiments/configs/final.json --fold 0 --run-name cv_final
```
"""),
    md("""### Interactive hyperparameter tuning

The next two cells are meant to be edited. Change any hyperparameter in the
first cell, then re-run both cells to train a small from-scratch model on CPU.
The demo uses 64px, 500 images per class and 4 epochs so it finishes in a few
minutes; it is a live tuning check, not a reproduction of the final 224px models.
"""),
    code("""# EDIT THESE HYPERPARAMETERS AND RE-RUN THIS CELL + THE NEXT ONE
tune_cfg = Config.from_json('../experiments/configs/final.json')
tune_cfg.train_csv = '../data/kaggle/kaggle/train.csv'
tune_cfg.train_dir = '../data/kaggle/kaggle'
tune_cfg.test_csv = '../data/Dataset/test_dataset.csv'
tune_cfg.test_dir = '../data/Dataset/testimages'
tune_cfg.image_size = 64          # try 64 / 96 / 128
tune_cfg.num_epochs = 4           # try 1-5
tune_cfg.warmup_epochs = 1
tune_cfg.batch_size = 64
tune_cfg.lr = 1e-3
tune_cfg.opt = 'adamw'            # 'adamw', 'sgd', 'momentum', 'adam', ...
tune_cfg.activation = 'relu'      # 'relu', 'gelu', 'silu', ...
tune_cfg.label_smoothing = 0.1
tune_cfg.limit_per_class = 500    # small subset for the CPU demo
tune_cfg.num_workers = 0
tune_cfg.device = 'cpu'
tune_cfg.ema = False              # too few steps for a useful EMA
tune_cfg.run_name = 'notebook_demo'
tune_cfg.output_dir = '../experiments'
tune_cfg
"""),
    code("""import shutil
from src.train import fit

shutil.rmtree('../experiments/notebook_demo', ignore_errors=True)
demo_model, demo_hist = fit(tune_cfg)
print(f"Demo validation accuracy (last epoch): "
      f"{100 * demo_hist['val_acc'][-1]:.2f}%")
"""),
    md("""The demo's accuracy is intentionally modest (64px, 1,000 images, 4 epochs).
It demonstrates that the hyperparameters are live; the submitted models use
the full dataset at 224px for 90 epochs with EMA.
"""),
    md("## 4. Hyperparameter sweep (160px, fold 0, 30 epochs)"),
    code("""rows = []
for path in sorted(glob.glob('../experiments/sweep_*/fold0_results.json')):
    r = json.load(open(path))
    c = r['config']
    rows.append({
        'run': r['run_name'],
        'params (M)': round(r['params'] / 1e6, 2),
        'opt': c['opt'],
        'lr': c['lr'],
        'label smoothing': c['label_smoothing'],
        'val acc (%)': round(100 * r['best_val_acc'], 2),
        'val F1 (%)': round(100 * r.get('best_val_f1', 0.0), 2),
        'best epoch': r['best_epoch'],
    })
sweep_df = pd.DataFrame(rows).sort_values('val acc (%)', ascending=False)
sweep_df.reset_index(drop=True)
"""),
    code("""display(Image(filename='../report/figures/sweep_comparison.png'))
display(Image(filename='../report/figures/sweep_best_history.png'))
"""),
    md("""**Observations.** Every convolutional configuration reaches >99.6%
validation accuracy; the from-scratch MLP baseline was 86.7%. Removing
augmentation produces the largest drop, confirming its regularizing role.
The medium ResNet-18-style model was selected for the final run because the
tiny 2.7M model's +0.02% margin is within noise and short-schedule models
tend to lose their edge at longer schedules.
"""),
    md("""## 5. Final 5-fold cross-validation (224px, 90 epochs)

The winning recipe is retrained on all five folds. Each fold supplies one
ensemble member, so every training image is covered while validation stays
clean.
"""),
    code("""cv_rows = []
for path in sorted(glob.glob('../experiments/cv_final/fold*_results.json')):
    r = json.load(open(path))
    cv_rows.append({
        'fold': r['fold'],
        'val acc (%)': round(100 * r['best_val_acc'], 5),
        'best epoch': r['best_epoch'],
        'EMA weights': r['used_ema'],
        'train images': r['train_size'],
        'val images': r['val_size'],
    })
cv_df = pd.DataFrame(cv_rows).sort_values('fold')
print(f"Mean fold accuracy: {cv_df['val acc (%)'].mean():.4f}%")
cv_df.reset_index(drop=True)
"""),
    code("""display(Image(filename='../report/figures/cv_curves.png'))
display(Image(filename='../report/figures/cv_scores.png'))
"""),
    md("""### Metric logging example

Every epoch appends a JSON line with train/validation loss, accuracy and F1
(the assignment's required logging).
"""),
    code("""with open('../experiments/cv_final/fold0_metrics.jsonl') as f:
    lines = f.readlines()
print('first epoch:', lines[0].strip())
print('last epoch: ', lines[-1].strip())
"""),
    md("""## 6. TTA + ensemble submission

Each of the five fold models predicts the test set under four deterministic
TTA views (identity, horizontal flip, two zoom levels). Probabilities are
averaged within a model and then across models. Only model outputs are used
for `submission.csv`.
"""),
    code("""summary = json.load(open('../experiments/cv_final/ensemble_probs_summary.json'))
summary
"""),
    code("""sub = pd.read_csv('../submission/submission_cv_tta.csv')
print(f"Rows: {len(sub)} | glasses: {int(sub['glasses'].sum())} | "
      f"no glasses: {int((1 - sub['glasses']).sum())}")
sub.head(10)
"""),
    md("""## 7. Model comparison and conclusion

| Model (from scratch) | Parameters | Validation accuracy |
|---|---|---|
| MLP, 64px, [512,256,128,64] (earlier baseline) | 6.5M | 86.69% |
| Residual CNN, 224px, 5-fold mean | 11.2M | **99.95%** |

- The convolutional inductive bias accounts for the large jump; no pretrained
  weights are involved.
- Augmentation, label smoothing, cosine scheduling and EMA stabilize and
  improve training.
- Five-fold cross-validation plus TTA ensembling reduces variance and yields
  the final test predictions.

**Kaggle test score: 0.99872** (approximately 7 errors out of 5,412 images).

**Compliance:** no transfer learning, no pretrained checkpoints, no ViT/DDPM.
The final `submission.csv` is generated exclusively from the from-scratch
network outputs.
"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.13"},
}

os.makedirs("notebooks", exist_ok=True)
path = "notebooks/glasses_classifier.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)
print(f"Wrote {path}")
