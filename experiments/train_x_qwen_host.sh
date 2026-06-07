#!/usr/bin/env bash
set -euo pipefail
python scripts/train_medlatent_x.py --config configs/medlatent_x.yaml "$@"
