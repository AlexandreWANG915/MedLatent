"""HuggingFace runtime for MedLatent-X.

MedLatent-X reuses frozen latent states produced by an encoder-family
MedLatent-H distiller, projects them into the host-family hidden space, and
trains only the projector plus host-side boundary embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache import FamilyLatentCache
from .hf_data import MedLatentDiagnosisDataset, collate_medlatent
from .hf_medlatent_h import (
    _assemble_blocks,
    _build_position_ids,
    _legacy_cache,
    _slice_last_positions,
    _to_dynamic_cache,
)
from .losses import diagnosis_cross_entropy
from .modules import BoundaryEmbeddings, LatentDistiller, LatentProjector


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.float32:
        return "float32"
    return str(dtype).replace("torch.", "")


def _load_tokenizer(model_name: str, *, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@torch.no_grad()
def _encoder_latent_hiddens(
    model,
    distiller: LatentDistiller,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    num_latents: int,
) -> torch.Tensor:
    """Return encoder-family latent hidden states shaped [B, K, H]."""

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
    )
    past = out.past_key_values
    prefix_mask = attention_mask
    current = distiller.begin_embedding(
        dtype=model.get_input_embeddings().weight.dtype,
        device=device,
    ).expand(input_ids.shape[0], -1, -1)

    hiddens: list[torch.Tensor] = []
    for _ in range(num_latents):
        prefix_mask = torch.cat([prefix_mask, torch.ones(input_ids.shape[0], 1, dtype=prefix_mask.dtype, device=device)], dim=1)
        position_ids = prefix_mask.sum(dim=1, keepdim=True) - 1
        latent_out = model(
            inputs_embeds=current,
            attention_mask=prefix_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
        past = latent_out.past_key_values
        hidden = latent_out.hidden_states[-1][:, -1, :]
        hiddens.append(hidden)
        current = distiller(hidden).unsqueeze(1)
    return torch.stack(hiddens, dim=1)


def build_family_cache_real(
    *,
    encoder_model_name: str,
    distiller_checkpoint: str | Path,
    train_file: str | Path,
    hospital_dir: str | Path,
    output_dir: str | Path,
    split: str,
    num_hospitals: int,
    num_latents: int,
    max_prompt_length: int,
    max_target_length: int,
    batch_size: int,
    max_samples: int | None,
    device: str,
    dtype_name: str,
    local_files_only: bool,
    hpo_embeddings_file: str | None,
    hpo_ic_file: str | None,
) -> dict[str, object]:
    dtype = _torch_dtype(dtype_name)
    resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(encoder_model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        encoder_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hidden_size = int(model.config.hidden_size)
    distiller = LatentDistiller.load(distiller_checkpoint, map_location=resolved_device).to(device=resolved_device, dtype=dtype)
    distiller.eval()
    if distiller.hidden_size != hidden_size:
        raise ValueError(f"Distiller hidden size {distiller.hidden_size} does not match encoder hidden size {hidden_size}")

    dataset = MedLatentDiagnosisDataset(
        data_file=train_file,
        hospital_dir=hospital_dir,
        tokenizer=tokenizer,
        num_hospitals=num_hospitals,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        hpo_embeddings_file=hpo_embeddings_file,
        hpo_ic_file=hpo_ic_file,
    )
    if max_samples is not None:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_medlatent(rows, pad_token_id=tokenizer.pad_token_id),
    )

    entries: dict[str, torch.Tensor] = {}
    for batch in loader:
        for hospital_id in range(num_hospitals):
            latent = _encoder_latent_hiddens(
                model,
                distiller,
                batch["hospital_ids_all"][hospital_id],
                batch["hospital_mask_all"][hospital_id],
                num_latents=num_latents,
            )
            latent = latent.detach().cpu().to(dtype)
            for row, case_id in enumerate(batch["case_ids"]):
                entries[f"{case_id}#{hospital_id}"] = latent[row]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_name = f"{split}.pt"
    torch.save(
        {
            "module_type": FamilyLatentCache.module_type,
            "split": split,
            "dtype": _dtype_name(dtype),
            "num_latents": num_latents,
            "encoder_dim": hidden_size,
            "entries": entries,
        },
        output_path / shard_name,
    )
    manifest = {
        "module_type": FamilyLatentCache.module_type,
        "encoder_model_name": encoder_model_name,
        "distiller_checkpoint": str(distiller_checkpoint),
        "num_latents": num_latents,
        "encoder_dim": hidden_size,
        "dtype": _dtype_name(dtype),
        "max_prompt_length": max_prompt_length,
        "num_hospitals": num_hospitals,
        "splits": {
            split: {
                "path": shard_name,
                "entries": len(entries),
                "data_file": str(train_file),
            }
        },
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"entries": len(entries), "cache_dir": str(output_path), "split": split}


def _projected_kv_block(model, projector: LatentProjector, boundary: BoundaryEmbeddings, latent_hidden: torch.Tensor):
    projected = projector(latent_hidden)
    wrapped = boundary.wrap(projected)
    mask = torch.ones(wrapped.shape[:2], dtype=torch.long, device=wrapped.device)
    out = model(inputs_embeds=wrapped, attention_mask=mask, use_cache=True, output_hidden_states=False)
    return _slice_last_positions(_legacy_cache(out.past_key_values), wrapped.shape[1])


def forward_medlatent_x_batch(model, projector: LatentProjector, boundary: BoundaryEmbeddings, cache: FamilyLatentCache, batch, *, split: str):
    device = next(model.parameters()).device
    dtype = model.get_input_embeddings().weight.dtype
    case_ids = list(batch["case_ids"])
    hospital_order = list(range(int(cache.manifest["num_hospitals"])))
    blocks_by_hospital = {}
    for hospital_id in hospital_order:
        latent_hidden = cache.lookup_batch(split, case_ids, hospital_id, device=device, dtype=dtype)
        blocks_by_hospital[hospital_id] = _projected_kv_block(model, projector, boundary, latent_hidden)

    legacy_cache, latent_mask = _assemble_blocks(blocks_by_hospital, hospital_order)
    host_question_ids = batch["host_question_ids"].to(device)
    host_question_mask = batch["host_question_mask"].to(device)
    host_attention = torch.cat([latent_mask.to(device), host_question_mask], dim=1)
    out_question = model(
        input_ids=host_question_ids,
        attention_mask=host_attention,
        position_ids=_build_position_ids(latent_mask.to(device), host_question_ids.shape[1]),
        past_key_values=_to_dynamic_cache(legacy_cache),
        use_cache=True,
        return_dict=True,
    )

    target_ids = batch["target_ids"].to(device)
    target_labels = batch["target_labels"].to(device)
    host_prefix = torch.cat([latent_mask.to(device), host_question_mask], dim=1)
    target_attention = torch.cat([host_prefix, torch.ones_like(target_ids)], dim=1)
    out_target = model(
        input_ids=target_ids,
        attention_mask=target_attention,
        position_ids=_build_position_ids(host_prefix, target_ids.shape[1]),
        past_key_values=out_question.past_key_values,
        use_cache=False,
        return_dict=True,
    )
    loss = diagnosis_cross_entropy(out_question.logits[:, -1, :], out_target.logits, target_labels)
    first_pred = out_question.logits[:, -1, :].argmax(dim=-1)
    first_gold = target_labels[:, 0]
    first_acc = (first_pred == first_gold).float().mean().item()
    return loss, {"loss_ce": float(loss.detach().cpu()), "first_token_acc": float(first_acc)}


def train_medlatent_x_real(
    *,
    host_model_name: str,
    cache_dir: str | Path,
    train_file: str | Path,
    hospital_dir: str | Path,
    output_dir: str | Path,
    num_hospitals: int,
    max_prompt_length: int,
    max_target_length: int,
    batch_size: int,
    max_samples: int | None,
    steps: int,
    learning_rate: float,
    device: str,
    dtype_name: str,
    local_files_only: bool,
    hpo_embeddings_file: str | None,
    hpo_ic_file: str | None,
) -> dict[str, object]:
    dtype = _torch_dtype(dtype_name)
    resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = _load_tokenizer(host_model_name, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        host_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cache = FamilyLatentCache(cache_dir)
    cache.validate({"num_hospitals": num_hospitals, "max_prompt_length": max_prompt_length})
    encoder_dim = int(cache.manifest["encoder_dim"])
    host_dim = int(model.config.hidden_size)
    projector = LatentProjector(encoder_dim=encoder_dim, host_dim=host_dim).to(device=resolved_device, dtype=dtype)
    boundary = BoundaryEmbeddings(hidden_size=host_dim).to(device=resolved_device, dtype=dtype)

    dataset = MedLatentDiagnosisDataset(
        data_file=train_file,
        hospital_dir=hospital_dir,
        tokenizer=tokenizer,
        num_hospitals=num_hospitals,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        limit=max_samples if max_samples is not None else -1,
        hpo_embeddings_file=hpo_embeddings_file,
        hpo_ic_file=hpo_ic_file,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_medlatent(rows, pad_token_id=tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(list(projector.parameters()) + list(boundary.parameters()), lr=learning_rate)

    last_loss = None
    last_stats: dict[str, float] = {}
    step = 0
    while step < steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = forward_medlatent_x_batch(model, projector, boundary, cache, batch, split="train")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(projector.parameters()) + list(boundary.parameters()), 1.0)
            optimizer.step()
            step += 1
            last_loss = float(loss.detach().cpu())
            last_stats = {key: float(value) for key, value in stats.items()}
            print(json.dumps({"event": "train_step", "step": step, **last_stats}), flush=True)
            if step >= steps:
                break

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    projector.save(output_path / "projector_final.pt")
    boundary.save(output_path / "boundary_final.pt")
    summary = {"steps": step, "loss_ce": last_loss, **last_stats}
    (output_path / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
