#!/bin/bash
#SBATCH --job-name=gs_ft288
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --deadline=2026-10-23T18:00:00
#SBATCH --output=slurm/logs/ft288_%j.out
#SBATCH --error=slurm/logs/ft288_%j.err

set -uo pipefail
cd /home2/home/chakilam_sai/acadss/dl/glasses-scratch
PY=/home2/home/chakilam_sai/acadss/dl/assignment/venv/bin/python
CONFIG=experiments/configs/finetune_288.json

echo "288px fine-tune started: $(date) | config=$CONFIG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for fold in 0 1 2 3 4; do
  echo ""
  echo "=== START fold $fold $(date) ==="
  $PY -m src.run --config "$CONFIG" --run-name ft288 --fold "$fold" \
    --seed $((142 + fold)) \
    --init-from "experiments/cv_final/fold${fold}_best.pt" \
    --resume \
    || echo "!!! FAILED fold $fold"
  echo "=== END fold $fold $(date) ==="
done

echo ""
echo "Fine-tune CV summary:"
$PY -m src.summarize --dir experiments/ft288 --pattern "*_results.json" \
  | tee experiments/ft288_summary.txt

echo ""
echo "10-model (224 + 288) TTA ensemble prediction:"
CKPTS="$(ls experiments/cv_final/fold*_best.pt experiments/ft288/fold*_best.pt)"
$PY -m src.predict \
  --checkpoints $CKPTS \
  --output submission/submission_cv_tta_ft288.csv \
  --tta-views 4 \
  --batch-size 128 \
  --num-workers 12 \
  --probs-out experiments/ensemble_224_ft288.pt \
  || echo "!!! prediction failed"

echo "288px fine-tune + prediction finished: $(date)"
