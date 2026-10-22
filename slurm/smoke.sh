#!/bin/bash
#SBATCH --job-name=gs_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=slurm/logs/smoke_%j.out
#SBATCH --error=slurm/logs/smoke_%j.err

set -euo pipefail
cd /home2/home/chakilam_sai/acadss/dl/glasses-scratch
PY=/home2/home/chakilam_sai/acadss/dl/assignment/venv/bin/python

echo "Smoke test started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
$PY -m src.run \
  --config experiments/configs/sweep_01_r18_adamw.json \
  --run-name smoke_r18 \
  --fold 0 \
  --limit-per-class 160 \
  --num-epochs 2 \
  --image-size 96 \
  --batch-size 64 \
  --num-workers 4
echo "Smoke test finished: $(date)"
