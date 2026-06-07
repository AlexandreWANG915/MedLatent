#!/usr/bin/env bash
set -euo pipefail
python scripts/build_latent_cache.py --config configs/medlatent_x.yaml --encoder_family mediphi "$@"
