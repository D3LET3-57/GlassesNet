# GlassesNet

Binary image classification: does the person in the image wear glasses?

**Author:** Chakilam Lalith Sai (Roll No. 24125009)  
**Repository:** https://github.com/D3LET3-57/GlassesNet

The model is a residual convolutional network implemented with PyTorch and
trained from random initialization on the provided dataset. The project
includes the training pipeline, a systematic hyperparameter study, five-fold
cross-validation, a multi-view TTA ensemble, the executed notebook, and a
verbose report.

## Results

| Model | Input | Parameters | Validation accuracy |
|---|---|---:|---:|
| MLP baseline | 64x64 | 6.46M | 86.69% |
| Residual CNN, 5-fold mean | 224x224 | 11.18M | **99.9510% +/- 0.0277%** |

- Mean validation F1: **99.9511%**
- Test score: **0.99872** (approximately 7 errors in 5,412 images)
- Ensemble: 5 fold models x 4 deterministic TTA views
- Submission: 2,984 glasses / 2,428 no-glasses; 5,378/5,412 unanimous

## Checkpoints

The five-fold best checkpoints (EMA weights) are available as a GitHub release:

https://github.com/D3LET3-57/GlassesNet/releases/tag/v1.0-checkpoints

## Architecture

ResNet-18-style network:

- 7x7 stem convolution (3 -> 64, stride 2), batch normalization, ReLU, 3x3 max-pool
- Four residual stages with widths `[64, 128, 256, 512]` and two BasicBlocks per stage
- Each BasicBlock: Conv3x3-BN-ReLU-Conv3x3-BN with a projection shortcut
- Global average pooling, dropout 0.2, linear layer to 2 logits
- **11,177,538** trainable parameters
- Kaiming-normal initialization

Training recipe: cross-entropy with label smoothing 0.1, AdamW (lr 1e-3,
weight decay 5e-4), 5 warmup epochs + cosine decay, AMP, gradient clipping,
EMA (0.999), 224x224 input, batch size 128, 90 epochs.

## Dataset

- Training: 24,499 images (12,288 glasses / 12,211 no-glasses), 512x512 RGB
- Test: 5,412 images
- Stratified five-fold split with a fixed split seed (42)
- Each fold trains on 19,598 images and validates on 4,901 (last fold:
  19,604/4,895)

Expected data layout:

```
data/
├── kaggle/kaggle/
│   ├── train.csv          # id,file,glasses
│   └── *.jpg
└── Dataset/
    ├── test_dataset.csv   # id,file
    └── testimages/*.jpg
```

In this working environment `data/` is a symlink to the original assignment
dataset and `venv/` is a symlink to the shared virtual environment. Both are
excluded from the release tarball; use local paths or recreate the environment.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── logs.md                       # experiment log
├── src/
│   ├── config.py                 # Config dataclass (all hyperparameters)
│   ├── dataset.py                # dataset, transforms, stratified folds
│   ├── cnn_model.py              # BasicBlock + GlassesCNN
│   ├── train.py                  # training loop, EMA, metrics, checkpoints
│   ├── run.py                    # single-run CLI
│   ├── predict.py                # TTA + multi-model ensemble inference
│   ├── summarize.py              # experiment summaries
│   └── plots.py                  # figures
├── scripts/
│   ├── pick_final.py             # choose the final config from the sweep
│   ├── make_hp_configs.py        # hyperparameter screening configs
│   ├── make_report_plots.py      # report figures
│   ├── build_notebook.py         # concise notebook
│   └── build_selfcontained_notebook.py
├── slurm/
│   ├── smoke.sh                  # tiny end-to-end check
│   ├── sweep.sh                  # full 160px sweep
│   ├── cv_train.sh               # 5-fold CV + TTA submission
│   ├── ft288.sh                  # 288px fine-tune ablation
│   └── predict.sh                # ensemble prediction
├── experiments/
│   ├── configs/                  # JSON configs
│   ├── sweep_01...06/            # full 160px/30-epoch sweep
│   ├── cv_final/                 # final 5-fold checkpoints/results
│   ├── hp_gpu/                   # 128px/20-epoch HP screening
│   └── ft288/                    # 288px fine-tune ablation
├── notebooks/
│   ├── glasses_classifier_selfcontained.ipynb
│   └── notebook_selfcontained.pdf
├── report/
│   ├── report_verbose.tex
│   ├── report_verbose.pdf        # includes the notebook PDF appendix
│   ├── figures/
│   └── figures_hp/
└── submission/
    └── submission.csv
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the dataset in the layout above, or adjust the paths in
`experiments/configs/*.json` (`train_csv`, `train_dir`, `test_csv`, `test_dir`).

## Training

Single fold:

```bash
python -m src.run \
  --config experiments/configs/auto_final.json \
  --fold 0 --run-name cv_final
```

Five-fold CV + ensemble submission:

```bash
sbatch slurm/cv_train.sh
```

The script trains all five folds, writes the summary, and runs the TTA
ensemble prediction to `submission/submission_cv_tta.csv`.

Other entry points:

```bash
sbatch slurm/smoke.sh      # quick smoke test
sbatch slurm/sweep.sh      # 160px/30-epoch sweep
sbatch slurm/ft288.sh      # 288px fine-tune ablation
```

Hyperparameter screening (used for the report figures):

```bash
python scripts/make_hp_configs.py
for cfg in experiments/configs/hp_gpu/*.json; do
  python -m src.run --config "$cfg" --fold 0 --seed 42
done
python scripts/make_report_plots.py
```

## Inference and submission

```bash
python -m src.predict \
  --checkpoint-dir experiments/cv_final \
  --pattern "fold*_best.pt" \
  --output submission/submission.csv \
  --tta-views 4 \
  --batch-size 192 \
  --num-workers 12
```

The shipped `submission/submission.csv` is the output of the five-fold
ensemble with four deterministic TTA views (identity, horizontal flip, and two
zoomed views). The file has 5,412 rows and columns `id,glasses`.

## Notebook and report

- **Notebook:** `notebooks/glasses_classifier_selfcontained.ipynb` — executed,
  32 cells, all model/training/inference code inline (no project-local imports).
  It includes an editable hyperparameter cell and a live CPU tuning demo.
  To run it outside this repository, set `GLASSES_DATA_ROOT` to the folder
  containing `kaggle/` and `Dataset/`.
- **Notebook PDF:** `notebooks/notebook_selfcontained.pdf`.
- **Report:** `report/report_verbose.pdf` (the executed notebook PDF is
  attached as Appendix B). LaTeX source: `report/report_verbose.tex`.
- **Experiment log:** `logs.md`.

## Experiment summary

| Experiment | Protocol | Result |
|---|---|---|
| Full sweep (`sweep_01`–`sweep_06`) | 160px, 30 epochs, fold 0 | 99.67–99.98% |
| HP screening | 128px, 20 epochs, full data, fold 0 | LR/activation/batch trends |
| Final CV (`cv_final`) | 224px, 90 epochs, 5 folds | 99.9510% +/- 0.0277% |
| 288px fine-tune (`ft288`) | 288px, 20 epochs, 5 folds | no change in mean; not used |

The final five-fold run took about two hours on one RTX 6000 Ada GPU.

## Notes

- All model, training, metric and inference computations use PyTorch.
- pandas/PIL handle data I/O and matplotlib produces the figures.
- Reproducibility: fold split seed 42, model seed 42 + fold index, deterministic
  TTA views.
- Checkpoint files (`*.pt`) are excluded from the source tarball; the five best
  fold checkpoints are in the GitHub release linked above.
