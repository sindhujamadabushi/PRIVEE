#!/bin/bash
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=t4_normal_q
#SBATCH --account=sindhuja

# set -euo pipefail


PROJECT_DIR=""
source PRIVEE/bin/activate
module load python3

SEED="${SLURM_ARRAY_TASK_ID}"

DATASET="CIFAR100"
ATTACK_STRENGTH="0.5"
DEFENSE="fh-ope"
RHO="0.1"
RESULT_DIR="$PROJECT_DIR/results/grna/${DATASET}/attack_strength_${ATTACK_STRENGTH}/${DEFENSE}"
LOG_DIR="$RESULT_DIR/logs"
METRIC_DIR="$RESULT_DIR/metrics"
LOG_FILE="$LOG_DIR/seed_${SEED}.log"
METRIC_FILE="$METRIC_DIR/seed_${SEED}.csv"

mkdir -p "$LOG_DIR" "$METRIC_DIR"
cd "$PROJECT_DIR"

{
    echo "=================================================="
    echo "SLURM job ID: ${SLURM_JOB_ID}"
    echo "SLURM array job ID: ${SLURM_ARRAY_JOB_ID}"
    echo "SLURM array task ID / seed: $SEED"
    echo "Host: $(hostname)"
    echo "Start time: $(date --iso-8601=seconds)"
    echo "Dataset: $DATASET"
    echo "Attack: grna"
    echo "Attack strength: $ATTACK_STRENGTH"
    echo "Defense: $DEFENSE"
    echo "Rho: $RHO"
    echo "=================================================="
} | tee "$LOG_FILE"

python -u main.py \
    --attack grna \
    --attack_strength "$ATTACK_STRENGTH" \
    --defense "$DEFENSE" \
    --decimals 2 \
    --rho "$RHO" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --epsilon 0.7\
    2>&1 | tee -a "$LOG_FILE"


from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
metric_path = Path(sys.argv[2])
seed = int(sys.argv[3])
text = log_path.read_text(encoding="utf-8", errors="replace")


def last_float(patterns: list[str], name: str) -> float:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    if not matches:
        raise RuntimeError(f"Could not extract {name} from {log_path}")
    return float(matches[-1])


accuracy_difference = last_float(
    [
        r"difference\s+in\s+accuracy\s*=\s*([-+0-9.eE]+)",
        r"difference\s+in\s+accuracy\s*:\s*([-+0-9.eE]+)\s*%?",
    ],
    "accuracy difference",
)
mse_without_defense = last_float(
    [r"(?:GRNA\s+)?MSE\s+without\s+defense\s*:\s*([-+0-9.eE]+)"],
    "MSE without defense",
)
mse_with_defense = last_float(
    [r"(?:GRNA\s+)?MSE\s+with\s+defense\s*:\s*([-+0-9.eE]+)"],
    "MSE with defense",
)

metric_path.parent.mkdir(parents=True, exist_ok=True)
with metric_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "seed",
        "accuracy_difference",
        "mse_without_defense",
        "mse_with_defense",
    ])
    writer.writerow([
        seed,
        accuracy_difference,
        mse_without_defense,
        mse_with_defense,
    ])

print(f"Saved metrics to {metric_path}")
PYMETRICS

echo "End time: $(date --iso-8601=seconds)" | tee -a "$LOG_FILE"
