from __future__ import annotations

import argparse
import os
from pathlib import Path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="", help="Path to a MedLatent YAML config template.")
    parser.add_argument("--dry_run", action="store_true", help="Validate arguments and print the intended action.")


def read_config_text(path: str) -> str:
    if not path:
        return ""
    text = Path(path).read_text()
    for key, value in os.environ.items():
        text = text.replace("${" + key + "}", value)
    return text


def dry_run_or_raise(*, name: str, config_text: str, dry_run: bool) -> None:
    if dry_run:
        print(f"{name}: dry run OK")
        print(config_text)
        return
    raise SystemExit(
        f"{name} requires connecting the CLI to the local model/data runtime. "
        "Core MedLatent modules are in medlatent/ and are covered by tests."
    )
