#!/bin/bash
#SBATCH --job-name=gs_pred
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/logs/predict_%j.out
#SBATCH --error=slurm/logs/predict_%j.err

set -euo pipefail
cd /home2/home/chakilam_sai/acadss/dl/glasses-scratch
PY=/home2/home/chakilam_sai/acadss/dl/assignment/venv/bin/python

echo "TTA ensemble started: $(date)"
$PY -m src.predict \
  --checkpoint-dir experiments/cv_final \
  --pattern "fold*_best.pt" \
  --output submission/submission_cv_tta.csv \
  --tta-views 4 \
  --batch-size 192 \
  --num-workers 12 \
  --probs-out experiments/cv_final/ensemble_probs.pt
echo "TTA ensemble finished: $(date)"
