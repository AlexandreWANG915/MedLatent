#!/usr/bin/env python
"""Build frozen encoder-family latent caches for MedLatent-X."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_common_args, dry_run_or_raise, read_config_text
from medlatent.hf_medlatent_x import build_family_cache_real


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--encoder_family", default="", help="Optional encoder family to cache.")
    parser.add_argument("--encoder_model_name", default="", help="Frozen encoder-family HuggingFace model path or id.")
    parser.add_argument("--distiller_checkpoint", default="", help="MedLatent-H distiller checkpoint for the encoder family.")
    parser.add_argument("--train_file", default="", help="Diagnosis JSON file used for this cache split.")
    parser.add_argument("--hospital_dir", default="", help="Directory containing hospital_*.json shards.")
    parser.add_argument("--output_dir", default="", help="Cache output directory.")
    parser.add_argument("--split", default="train", help="Cache split name.")
    parser.add_argument("--num_hospitals", type=int, default=3)
    parser.add_argument("--num_latents", type=int, default=32)
    parser.add_argument("--max_prompt_length", type=int, default=320)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--hpo_embeddings_file", default="", help="Optional HPO embedding JSON for IC-weighted cosine retrieval.")
    parser.add_argument("--hpo_ic_file", default="", help="Optional HPO information-content JSON for IC-weighted cosine retrieval.")
    args = parser.parse_args()
    real_args = [args.encoder_model_name, args.distiller_checkpoint, args.train_file, args.hospital_dir, args.output_dir]
    if args.dry_run or not any(real_args):
        dry_run_or_raise(name="build_latent_cache", config_text=read_config_text(args.config), dry_run=args.dry_run)
        return
    missing = [name for name, value in {
        "encoder_model_name": args.encoder_model_name,
        "distiller_checkpoint": args.distiller_checkpoint,
        "train_file": args.train_file,
        "hospital_dir": args.hospital_dir,
        "output_dir": args.output_dir,
    }.items() if not value]
    if missing:
        parser.error(f"real cache build requires: {', '.join(missing)}")
    summary = build_family_cache_real(
        encoder_model_name=args.encoder_model_name,
        distiller_checkpoint=args.distiller_checkpoint,
        train_file=args.train_file,
        hospital_dir=args.hospital_dir,
        output_dir=args.output_dir,
        split=args.split,
        num_hospitals=args.num_hospitals,
        num_latents=args.num_latents,
        max_prompt_length=args.max_prompt_length,
        max_target_length=args.max_target_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device=args.device,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        hpo_embeddings_file=args.hpo_embeddings_file or None,
        hpo_ic_file=args.hpo_ic_file or None,
    )
    print(f"cache build complete: cache_dir={summary['cache_dir']} split={summary['split']} entries={summary['entries']}")


if __name__ == "__main__":
    main()
