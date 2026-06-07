#!/usr/bin/env python
"""Train MedLatent-X cross-family latent projectors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_common_args, dry_run_or_raise, read_config_text
from medlatent.hf_medlatent_x import train_medlatent_x_real


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--host_model_name", default="", help="Frozen host-family HuggingFace model path or id.")
    parser.add_argument("--cache_dir", default="", help="MedLatent-X latent cache directory.")
    parser.add_argument("--train_file", default="", help="Diagnosis train JSON file.")
    parser.add_argument("--hospital_dir", default="", help="Directory containing hospital_*.json shards.")
    parser.add_argument("--output_dir", default="", help="Training output directory.")
    parser.add_argument("--num_hospitals", type=int, default=3)
    parser.add_argument("--max_prompt_length", type=int, default=320)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional data cap; useful for cache/training smoke tests.")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--hpo_embeddings_file", default="", help="Optional HPO embedding JSON for IC-weighted cosine retrieval.")
    parser.add_argument("--hpo_ic_file", default="", help="Optional HPO information-content JSON for IC-weighted cosine retrieval.")
    args = parser.parse_args()
    real_args = [args.host_model_name, args.cache_dir, args.train_file, args.hospital_dir, args.output_dir]
    if args.dry_run or not any(real_args):
        dry_run_or_raise(name="train_medlatent_x", config_text=read_config_text(args.config), dry_run=args.dry_run)
        return
    missing = [name for name, value in {
        "host_model_name": args.host_model_name,
        "cache_dir": args.cache_dir,
        "train_file": args.train_file,
        "hospital_dir": args.hospital_dir,
        "output_dir": args.output_dir,
    }.items() if not value]
    if missing:
        parser.error(f"real training requires: {', '.join(missing)}")
    summary = train_medlatent_x_real(
        host_model_name=args.host_model_name,
        cache_dir=args.cache_dir,
        train_file=args.train_file,
        hospital_dir=args.hospital_dir,
        output_dir=args.output_dir,
        num_hospitals=args.num_hospitals,
        max_prompt_length=args.max_prompt_length,
        max_target_length=args.max_target_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        steps=args.steps,
        learning_rate=args.learning_rate,
        device=args.device,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        hpo_embeddings_file=args.hpo_embeddings_file or None,
        hpo_ic_file=args.hpo_ic_file or None,
    )
    print(f"real MedLatent-X training complete: output_dir={args.output_dir} steps={summary['steps']}")


if __name__ == "__main__":
    main()
