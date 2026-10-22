#!/bin/bash
#SBATCH --job-name=gs_cv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/cv_%j.out
#SBATCH --error=slurm/logs/cv_%j.err

set -uo pipefail
cd /home2/home/chakilam_sai/acadss/dl/glasses-scratch
PY=/home2/home/chakilam_sai/acadss/dl/assignment/venv/bin/python

if [ $# -ge 1 ]; then
  CONFIG=$1
else
  $PY scripts/pick_final.py || true
  CONFIG=experiments/configs/auto_final.json
  [ -f "$CONFIG" ] || CONFIG=experiments/configs/final.json
fi

echo "5-fold CV started: $(date) | config=$CONFIG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for fold in 0 1 2 3 4; do
  echo ""
  echo "=== START fold $fold $(date) ==="
  $PY -m src.run --config "$CONFIG" --run-name cv_final --fold "$fold" \
    --seed $((42 + fold)) --resume \
    || echo "!!! FAILED fold $fold"
  echo "=== END fold $fold $(date) ==="
done

echo ""
echo "CV summary:"
$PY -m src.summarize --dir experiments/cv_final --pattern "*_results.json" \
  | tee experiments/cv_final_summary.txt

echo ""
echo "TTA ensemble prediction:"
$PY -m src.predict \
  --checkpoint-dir experiments/cv_final \
  --pattern "fold*_best.pt" \
  --output submission/submission_cv_tta.csv \
  --tta-views 4 \
  --batch-size 192 \
  --num-workers 12 \
  --probs-out experiments/cv_final/ensemble_probs.pt \
  || echo "!!! prediction failed"

echo "5-fold CV + prediction finished: $(date)"
