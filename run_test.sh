#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT="PPO"
PARAMS="${SCRIPT_DIR}/BEST_AGENTS/malware-train-ember-v0/PPO/params_2/params.json"
CHECKPOINT="${SCRIPT_DIR}/BEST_AGENTS/malware-train-ember-v0/PPO/params_2/checkpoint_000852/checkpoint-852"

python "${SCRIPT_DIR}/RAY_test_from_checkpoint.py" \
    --agent="${AGENT}" \
    --params="${PARAMS}" \
    --checkpoint="${CHECKPOINT}" \
    --save-files=True \
    "$@"
