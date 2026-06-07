#!/usr/bin/env python
"""Evaluate reconstruction leakage from transmitted MedLatent objects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_common_args, dry_run_or_raise, read_config_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--method", choices=["medlatent_h", "medlatent_x"], default="medlatent_h")
    args = parser.parse_args()
    dry_run_or_raise(name="evaluate_leakage", config_text=read_config_text(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
