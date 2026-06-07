#!/usr/bin/env bash
set -euo pipefail
python scripts/train_medlatent_h.py --config configs/medlatent_h.yaml "$@"
