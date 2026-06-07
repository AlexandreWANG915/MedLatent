#!/usr/bin/env bash
set -euo pipefail
python scripts/evaluate_diagnosis.py --config configs/eval.yaml "$@"
