"""OxygenREC-v2 GPU/NPU固定输入对齐的共享执行逻辑。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, fields
import hashlib
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import torch

from .device import max_memory_allocated, reset_peak_memory_stats, synchronize
from .data import (
    Split,
    TemporalBoundaries,
    build_daily_listwise_samples,
    build_listwise_sid_model_batch,
    load_retailrocket_events,
)
from .model import OxygenRECConfig, OxygenRECModel
from .sid import PrefixTrie, SIDRegistry


REFERENCE_SCHEMA_VERSION = 1
REFERENCE_PROTOCOL = "oxygenrec_v2_full_fixed_batch_fp32"
ALIGNMENT_SOURCE_FILES = (
    "run_v2_device_alignment.sh",
    "scripts/run_v2_device_probe.py",
    "scripts/compare_v2_device_probes.py",
    "scripts/export_v2_gpu_reference.py",
    "scripts/compare_device_reference.py",
    "src/oxygenrec/device.py",
    "src/oxygenrec/migration_alignment.py",
    "src/oxygenrec/model.py",
    "src/oxygenrec/data/model_inputs.py",
    "src/oxygenrec/data/temporal.py",
    "src/oxygenrec/sid.py",
)
BEHAVIOR_WEIGHTS = (1.2, 1.5, 2.0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file_hashes(project_root: Path) -> dict[str, str]:
    """Fingerprint the exact source files that define the alignment protocol."""

    return {
        relative_path: file_sha256(project_root / relative_path)
        for relative_path in ALIGNMENT_SOURCE_FILES
    }


def git_state(project_root: Path) -> dict[str, object]:
    """Record the code revision without treating ignored experiment assets as source."""

    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        revision = run("rev-parse", "HEAD")
        tracked_status = run("status", "--porcelain", "--untracked-files=no")
        untracked = run("ls-files", "--others", "--exclude-standard")
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "tracked_dirty": None,
            "untracked_file_count": None,
        }
    return {
        "commit": revision,
        "tracked_dirty": bool(tracked_status),
        "untracked_file_count": len(untracked.splitlines()) if untracked else 0,
    }


def build_fixed_batch(
    events_path: Path,
    checkpoint: dict[str, Any],
    registry: SIDRegistry,
    *,
    samples: int,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Rebuild the deterministic validation batch shared by both devices."""

    config = restored_config(checkpoint)
    checkpoint_args = checkpoint.get("args", {})
    seed = int(checkpoint_args.get("seed", 17))
    train_limit = int(checkpoint_args.get("max_train_samples", 5_000))
    boundaries = TemporalBoundaries(**checkpoint["boundaries"])
    events = [
        event
        for event in load_retailrocket_events(events_path)
        if event.item_id in registry.item_to_sid
    ]
    rows = build_daily_listwise_samples(
        events,
        boundaries,
        list_size=config.max_target_items,
        max_history=config.max_history_items,
        max_samples_per_split={
            Split.TRAIN: train_limit,
            Split.VALIDATION: samples,
            Split.TEST: 1,
        },
        sample_seed=seed,
    )
    by_split: dict[Split, list[Any]] = defaultdict(list)
    for row in rows:
        by_split[row.split].append(row)
    validation = by_split[Split.VALIDATION]
    if len(validation) != samples:
        raise RuntimeError(f"expected {samples} validation lists, got {len(validation)}")
    raw = build_listwise_sid_model_batch(
        validation, registry, max_history_items=config.max_history_items
    )
    behavior_ids = torch.tensor(raw.target_behavior_ids, dtype=torch.long)
    sid_tokens = config.max_target_items * config.sid_levels
    behavior_weights = torch.tensor(BEHAVIOR_WEIGHTS, dtype=torch.float32)
    batch = {
        "history_sids": torch.tensor(raw.history_sids, dtype=torch.long),
        "history_padding_mask": torch.tensor(
            raw.history_padding_mask, dtype=torch.bool
        ),
        "history_behavior_ids": torch.tensor(
            raw.history_behavior_ids, dtype=torch.long
        ),
        "target_sids": torch.tensor(raw.target_sids, dtype=torch.long),
        "behavior_instruction_ids": behavior_ids,
        "token_weights": behavior_weights[behavior_ids]
        .unsqueeze(1)
        .expand(-1, sid_tokens)
        .clone(),
    }
    metadata = {
        "sample_seed": seed,
        "validation_samples": samples,
        "list_size": config.max_target_items,
        "max_history_items": config.max_history_items,
        "boundaries": asdict(boundaries),
    }
    return batch, metadata


def restored_config(checkpoint: Mapping[str, Any]) -> OxygenRECConfig:
    known = {field.name for field in fields(OxygenRECConfig)}
    values = {
        key: value
        for key, value in checkpoint["model_config"].items()
        if key in known
    }
    config = OxygenRECConfig(**values)
    if config.behavior_vocab_size < 1 or config.behavior_instruction_vocab_size < 1:
        raise ValueError("v2 Full alignment requires history and target behavior inputs")
    if config.max_target_items < 1:
        raise ValueError("v2 Full alignment requires listwise targets")
    return config


def move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def forward_kwargs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "history_sids": batch["history_sids"],
        "history_padding_mask": batch["history_padding_mask"],
        "history_behavior_ids": batch["history_behavior_ids"],
        "target_sids": batch["target_sids"],
        "behavior_instruction_ids": batch["behavior_instruction_ids"],
        "token_weights": batch["token_weights"],
    }


def generation_kwargs(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "history_behavior_ids": batch["history_behavior_ids"],
        "behavior_instruction_ids": batch["behavior_instruction_ids"],
    }


def load_model(
    checkpoint: Mapping[str, Any], config: OxygenRECConfig, device: torch.device
) -> OxygenRECModel:
    model = OxygenRECModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def inference_outputs(
    model: OxygenRECModel,
    batch: Mapping[str, torch.Tensor],
    trie: PrefixTrie,
    *,
    beam_width: int,
    output_items: int,
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        output = model(**forward_kwargs(batch))
        greedy = model.generate(
            batch["history_sids"],
            batch["history_padding_mask"],
            trie,
            output_items=output_items,
            **generation_kwargs(batch),
        )
        beam = model.beam_search(
            batch["history_sids"],
            batch["history_padding_mask"],
            trie,
            beam_width=beam_width,
            output_items=output_items,
            **generation_kwargs(batch),
        )
    return {
        "logits": tuple(tensor.detach().cpu() for tensor in output.logits),
        "loss": output.loss.detach().cpu(),
        "ntp_loss": output.ntp_loss.detach().cpu(),
        "level_losses": tuple(
            tensor.detach().cpu() for tensor in output.level_losses
        ),
        "greedy_sids": greedy.detach().cpu(),
        "beam_sids": beam.semantic_ids.detach().cpu(),
        "beam_scores": beam.scores.detach().cpu(),
    }


def deterministic_train_step(
    model: OxygenRECModel,
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    """在eval模式做确定性反向和一次AdamW更新；不引入dropout随机差异。"""

    model.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, foreach=False, fused=False
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer.zero_grad(set_to_none=True)
    output = model(**forward_kwargs(batch))
    output.loss.backward()
    synchronize(device)
    gradients = {
        name: parameter.grad.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None
    ]
    if not gradients:
        raise RuntimeError("fixed-batch backward produced no gradients")
    if not all(torch.isfinite(tensor).all() for tensor in gradients.values()):
        raise RuntimeError("fixed-batch backward produced non-finite gradients")
    optimizer.step()
    synchronize(device)
    parameter_deltas = {
        name: (parameter.detach() - before[name]).cpu()
        for name, parameter in model.named_parameters()
    }
    max_delta = max(
        float(tensor.abs().max()) for tensor in parameter_deltas.values()
    )
    if max_delta <= 0.0:
        raise RuntimeError("fixed-batch AdamW step did not update parameters")
    return {
        "learning_rate": learning_rate,
        "loss": output.loss.detach().cpu(),
        "gradients": gradients,
        "missing_gradients": missing_gradients,
        "parameter_deltas": parameter_deltas,
        "max_parameter_delta": max_delta,
    }


def measure_forward(
    model: OxygenRECModel,
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    warmup_steps: int,
    timed_steps: int,
) -> dict[str, Any]:
    if warmup_steps < 0 or timed_steps < 1:
        raise ValueError("warmup_steps must be non-negative and timed_steps positive")
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup_steps):
            model(**forward_kwargs(batch))
        synchronize(device)
        memory_supported = reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(timed_steps):
            model(**forward_kwargs(batch))
        synchronize(device)
        elapsed = time.perf_counter() - started
    batch_size = int(batch["history_sids"].shape[0])
    return {
        "scope": "eval_forward_only",
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "elapsed_seconds": elapsed,
        "mean_step_seconds": elapsed / timed_steps,
        "samples_per_second": batch_size * timed_steps / elapsed,
        "peak_memory_allocated_bytes": (
            max_memory_allocated(device) if memory_supported else None
        ),
    }


def compare_tensor_maps(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    expected_names = set(expected)
    actual_names = set(actual)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    rows = {}
    all_close = not missing and not unexpected
    maximum = 0.0
    for name in sorted(expected_names & actual_names):
        left = expected[name]
        right = actual[name]
        same_shape = left.shape == right.shape
        if same_shape and left.numel():
            delta = (left - right).abs()
            max_abs = float(delta.max())
            mean_abs = float(delta.mean())
            close = bool(torch.allclose(left, right, atol=atol, rtol=rtol))
        elif same_shape:
            max_abs = 0.0
            mean_abs = 0.0
            close = True
        else:
            max_abs = float("inf")
            mean_abs = float("inf")
            close = False
        maximum = max(maximum, max_abs)
        all_close = all_close and close
        rows[name] = {
            "shape": list(right.shape),
            "same_shape": same_shape,
            "close": close,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
        }
    return {
        "all_close": all_close,
        "max_abs": maximum,
        "missing": missing,
        "unexpected": unexpected,
        "tensors": rows,
    }