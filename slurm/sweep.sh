#!/bin/bash
#SBATCH --job-name=gs_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=slurm/logs/sweep_%j.out
#SBATCH --error=slurm/logs/sweep_%j.err

set -uo pipefail
cd /home2/home/chakilam_sai/acadss/dl/glasses-scratch
PY=/home2/home/chakilam_sai/acadss/dl/assignment/venv/bin/python

echo "Sweep started: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

for cfg in experiments/configs/sweep_*.json; do
  name=$(basename "$cfg" .json)
  echo ""
  echo "=== START $name $(date) ==="
  $PY -m src.run --config "$cfg" --run-name "$name" --fold 0 --resume \
    || echo "!!! FAILED $name"
  echo "=== END $name $(date) ==="
done

echo ""
echo "Sweep summary:"
$PY -m src.summarize --dir experiments --pattern "*_results.json" \
  | tee experiments/sweep_summary.txt

echo "Sweep finished: $(date)"
