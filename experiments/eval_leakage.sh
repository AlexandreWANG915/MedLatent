#!/usr/bin/env bash
set -euo pipefail
python scripts/evaluate_leakage.py --config configs/eval.yaml "$@"
