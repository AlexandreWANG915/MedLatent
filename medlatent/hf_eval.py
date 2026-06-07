"""Real HuggingFace diagnosis inference for MedLatent-H/X."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache import FamilyLatentCache
from .hf_data import MedLatentDiagnosisDataset, collate_medlatent
from .hf_medlatent_h import (
    _append_embedding,
    _assemble_blocks,
    _build_position_ids,
    _slice_last_positions,
    _to_dynamic_cache,
)
from .hf_medlatent_x import _projected_kv_block
from .metrics import exact_match, token_f1
from .modules import BoundaryEmbeddings, LatentDistiller, LatentProjector


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _load_model_and_tokenizer(model_name: str, *, dtype: torch.dtype, device: torch.device, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def _medlatent_h_prefix(model, distiller: LatentDistiller, boundary: BoundaryEmbeddings, batch: dict, *, num_latents: int, device):
    hospital_order = list(range(len(batch["hospital_ids_all"])))
    hospital_blocks: dict[int, tuple] = {}
    block_len = int(num_latents) + 2
    for hospital_id in hospital_order:
        input_ids = batch["hospital_ids_all"][hospital_id].to(device)
        attention_mask = batch["hospital_mask_all"][hospital_id].to(device)
        prompt_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = prompt_out.past_key_values
        lengths = attention_mask.sum(dim=1) - 1
        hidden = prompt_out.hidden_states[-1][torch.arange(input_ids.shape[0], device=device), lengths, :]
        prefix_mask = attention_mask

        past_key_values, prefix_mask = _append_embedding(model, boundary.begin, prefix_mask, past_key_values)
        for step in range(int(num_latents)):
            if step == 0:
                step_input = distiller.begin_embedding(dtype=model.get_input_embeddings().weight.dtype, device=device)
                step_input = step_input.expand(input_ids.shape[0], 1, -1)
            else:
                step_input = distiller(hidden).unsqueeze(1)
            attention = torch.cat([prefix_mask, prefix_mask.new_ones((input_ids.shape[0], 1))], dim=1)
            latent_out = model(
                inputs_embeds=step_input,
                attention_mask=attention,
                position_ids=_build_position_ids(prefix_mask, 1),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = latent_out.past_key_values
            hidden = latent_out.hidden_states[-1][:, -1, :]
            prefix_mask = attention
        past_key_values, _ = _append_embedding(model, boundary.end, prefix_mask, past_key_values)
        hospital_blocks[hospital_id] = _slice_last_positions(past_key_values, block_len)

    legacy_cache, latent_mask = _assemble_blocks(hospital_blocks, hospital_order)
    host_ids = batch["host_question_ids"].to(device)
    host_mask = batch["host_question_mask"].to(device)
    host_attention = torch.cat([latent_mask.to(device), host_mask], dim=1)
    host_out = model(
        input_ids=host_ids,
        attention_mask=host_attention,
        position_ids=_build_position_ids(latent_mask.to(device), host_ids.shape[1]),
        past_key_values=_to_dynamic_cache(legacy_cache),
        use_cache=True,
        return_dict=True,
    )
    return host_out, host_attention


def _medlatent_x_prefix(model, projector: LatentProjector, boundary: BoundaryEmbeddings, cache: FamilyLatentCache, batch: dict, *, device):
    hospital_order = list(range(int(cache.manifest["num_hospitals"])))
    blocks_by_hospital = {}
    case_ids = list(batch["case_ids"])
    dtype = model.get_input_embeddings().weight.dtype
    for hospital_id in hospital_order:
        latent_hidden = cache.lookup_batch("train", case_ids, hospital_id, device=device, dtype=dtype)
        blocks_by_hospital[hospital_id] = _projected_kv_block(model, projector, boundary, latent_hidden)
    legacy_cache, latent_mask = _assemble_blocks(blocks_by_hospital, hospital_order)
    host_ids = batch["host_question_ids"].to(device)
    host_mask = batch["host_question_mask"].to(device)
    host_attention = torch.cat([latent_mask.to(device), host_mask], dim=1)
    host_out = model(
        input_ids=host_ids,
        attention_mask=host_attention,
        position_ids=_build_position_ids(latent_mask.to(device), host_ids.shape[1]),
        past_key_values=_to_dynamic_cache(legacy_cache),
        use_cache=True,
        return_dict=True,
    )
    return host_out, host_attention


@torch.no_grad()
def _greedy_decode(model, tokenizer, host_out, host_attention: torch.Tensor, *, max_new_tokens: int) -> list[int]:
    if host_attention.shape[0] != 1:
        raise ValueError("Real evaluation currently expects batch_size=1")
    generated: list[int] = []
    past_key_values = host_out.past_key_values
    attention = host_attention
    next_token = host_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    eos_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        if eos_id is not None and token_id == eos_id:
            break
        attention = torch.cat([attention, attention.new_ones((1, 1))], dim=1)
        position_ids = attention.sum(dim=1, keepdim=True) - 1
        out = model(
            input_ids=next_token,
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return generated


def evaluate_medlatent_real(
    *,
    method: str,
    model_name: str,
    checkpoint_dir: str | Path,
    data_file: str | Path,
    hospital_dir: str | Path,
    output: str | Path,
    cache_dir: str | Path | None,
    num_hospitals: int,
    num_latents: int,
    max_prompt_length: int,
    max_target_length: int,
    max_new_tokens: int,
    max_samples: int,
    device: str,
    dtype_name: str,
    local_files_only: bool,
    hpo_embeddings_file: str | None,
    hpo_ic_file: str | None,
) -> dict[str, float]:
    dtype = _torch_dtype(dtype_name)
    resolved_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_model_and_tokenizer(model_name, dtype=dtype, device=resolved_device, local_files_only=local_files_only)
    checkpoint_path = Path(checkpoint_dir)

    distiller = boundary = projector = cache = None
    if method == "medlatent_h":
        distiller = LatentDistiller.load(checkpoint_path / "distiller_final.pt", map_location=resolved_device).to(device=resolved_device, dtype=dtype)
        boundary = BoundaryEmbeddings.load(checkpoint_path / "boundary_final.pt", map_location=resolved_device).to(device=resolved_device, dtype=dtype)
        distiller.eval()
        boundary.eval()
    elif method == "medlatent_x":
        if cache_dir is None:
            raise ValueError("--cache_dir is required for medlatent_x")
        projector = LatentProjector.load(checkpoint_path / "projector_final.pt", map_location=resolved_device).to(device=resolved_device, dtype=dtype)
        boundary = BoundaryEmbeddings.load(checkpoint_path / "boundary_final.pt", map_location=resolved_device).to(device=resolved_device, dtype=dtype)
        cache = FamilyLatentCache(cache_dir)
        projector.eval()
        boundary.eval()
    else:
        raise ValueError(f"Unknown method: {method}")

    dataset = MedLatentDiagnosisDataset(
        data_file=data_file,
        hospital_dir=hospital_dir,
        tokenizer=tokenizer,
        num_hospitals=num_hospitals,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        limit=max_samples,
        hpo_embeddings_file=hpo_embeddings_file,
        hpo_ic_file=hpo_ic_file,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda rows: collate_medlatent(rows, pad_token_id=tokenizer.pad_token_id),
    )

    rows = []
    for batch in loader:
        if method == "medlatent_h":
            host_out, host_attention = _medlatent_h_prefix(model, distiller, boundary, batch, num_latents=num_latents, device=resolved_device)
        else:
            host_out, host_attention = _medlatent_x_prefix(model, projector, boundary, cache, batch, device=resolved_device)
        token_ids = _greedy_decode(model, tokenizer, host_out, host_attention, max_new_tokens=max_new_tokens)
        prediction = tokenizer.decode(token_ids, skip_special_tokens=True)
        reference = tokenizer.decode(batch["target_ids"][0], skip_special_tokens=True)
        rows.append(
            {
                "case_id": batch["case_ids"][0],
                "prediction": prediction,
                "reference": reference,
                "exact_match": exact_match(prediction, reference),
                "token_f1": token_f1(prediction, reference),
            }
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))
    accuracy = sum(1 for row in rows if row["exact_match"]) / len(rows) if rows else 0.0
    avg_token_f1 = sum(float(row["token_f1"]) for row in rows) / len(rows) if rows else 0.0
    metrics = {"accuracy": accuracy, "token_f1": avg_token_f1, "num_samples": float(len(rows))}
    metrics_path = output_path.with_suffix(output_path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
