#!/usr/bin/env python3
"""Continue v2 Full on the public-data training split, one device and full epochs."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.data import (
    Split, TemporalBoundaries, build_daily_listwise_samples, build_listwise_sid_model_batch,
)
from oxygenrec.data.events import load_retailrocket_events
from oxygenrec.device import device_name, max_memory_allocated, reset_peak_memory_stats
from oxygenrec.device import resolve_device, seed_torch, synchronize
from oxygenrec.migration_alignment import (
    BEHAVIOR_WEIGHTS, file_sha256, git_state, load_model, restored_config,
)
from oxygenrec.sid import SIDRegistry


def make_full_batch(samples, registry, config, device) -> dict[str, torch.Tensor]:
    raw = build_listwise_sid_model_batch(
        samples, registry, max_history_items=config.max_history_items
    )
    behavior_ids = torch.tensor(
        raw.target_behavior_ids, dtype=torch.long, device=device
    )
    weights = torch.tensor(BEHAVIOR_WEIGHTS, dtype=torch.float32, device=device)
    sid_tokens = config.max_target_items * config.sid_levels
    return {
        "history_sids": torch.tensor(raw.history_sids, dtype=torch.long, device=device),
        "history_padding_mask": torch.tensor(
            raw.history_padding_mask, dtype=torch.bool, device=device
        ),
        "history_behavior_ids": torch.tensor(
            raw.history_behavior_ids, dtype=torch.long, device=device
        ),
        "target_sids": torch.tensor(raw.target_sids, dtype=torch.long, device=device),
        "behavior_instruction_ids": behavior_ids,
        "token_weights": weights[behavior_ids]
        .unsqueeze(1)
        .expand(-1, sid_tokens),
    }


def cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(child) for child in value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("gpu", "npu"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1, help="additional full epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-train-samples", type=int,
        help="optional capped scale-up; omit to use the complete public-data train split",
    )
    parser.add_argument("--learning-rate", type=float)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.max_train_samples is not None and args.max_train_samples < 1:
        raise ValueError("max-train-samples must be positive")
    missing = [
        str(path) for path in (args.events, args.checkpoint, args.sid_registry)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing full-training inputs: " + ", ".join(missing))

    device = resolve_device(args.device)
    expected_type = "cuda" if args.platform == "gpu" else "npu"
    if device.type != expected_type:
        raise ValueError(f"platform {args.platform!r} requires {expected_type!r}")
    source = git_state(Path(__file__).resolve().parents[1])
    if source["commit"] is None or source["tracked_dirty"]:
        raise RuntimeError("full training requires a known commit and clean tracked files")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "full":
        raise ValueError("full training requires a v2 Full checkpoint")
    registry = SIDRegistry.from_json(args.sid_registry)
    if registry.version != checkpoint.get("sid_registry_version"):
        raise ValueError("checkpoint and SID registry versions differ")
    config = restored_config(checkpoint)
    boundaries = TemporalBoundaries(**checkpoint["boundaries"])
    original_args = checkpoint.get("args", {})
    prior_run = checkpoint.get("full_training", {})
    seed = int(original_args.get("seed", 17))
    learning_rate = (
        args.learning_rate if args.learning_rate is not None
        else float(prior_run.get("learning_rate", original_args.get("learning_rate", 3e-4)))
    )
    if learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    completed_epochs = int(prior_run.get("completed_epochs", 0))
    global_steps = int(prior_run.get("global_steps", 0))
    if prior_run and prior_run.get("precision") != args.precision:
        raise ValueError("resume precision differs from prior full-training run")
    if prior_run and prior_run.get("max_train_samples") != args.max_train_samples:
        raise ValueError("resume train-sample cap differs from prior full-training run")
    if prior_run and prior_run.get("batch_size") != args.batch_size:
        raise ValueError("resume batch-size differs from prior full-training run")
    if prior_run and not checkpoint.get("optimizer_state"):
        raise ValueError("resume checkpoint has no optimizer state")

    print("stage=build_complete_v2_train_split", flush=True)
    events = [
        event for event in load_retailrocket_events(args.events)
        if event.item_id in registry.item_to_sid
    ]
    limits = {Split.VALIDATION: 1, Split.TEST: 1}
    if args.max_train_samples is not None:
        limits[Split.TRAIN] = args.max_train_samples
    rows = build_daily_listwise_samples(
        events,
        boundaries,
        list_size=config.max_target_items,
        max_history=config.max_history_items,
        max_samples_per_split=limits,
        sample_seed=seed,
    )
    train_samples = [row for row in rows if row.split is Split.TRAIN]
    if not train_samples:
        raise RuntimeError("complete v2 train split is empty")
    behavior_counts = Counter(row.target_behavior.value for row in train_samples)
    print(
        f"stage=train_split_ready samples={len(train_samples)} "
        f"cap={args.max_train_samples} behaviors={dict(behavior_counts)}",
        flush=True,
    )

    seed_torch(seed, device)
    model = load_model(checkpoint, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer_loaded = bool(checkpoint.get("optimizer_state"))
    if optimizer_loaded:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "events_sha256": file_sha256(args.events),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "sid_registry_sha256": file_sha256(args.sid_registry),
    }
    for name in ("events_sha256", "sid_registry_sha256"):
        if prior_run and prior_run.get(name) != input_hashes[name]:
            raise ValueError(f"resume input differs in {name}")
    for additional_epoch in range(1, args.epochs + 1):
        epoch = completed_epochs + additional_epoch
        checkpoint_path = args.output_dir / f"v2_full_{args.platform}_{args.precision}_epoch{epoch}.pt"
        summary_path = args.output_dir / f"v2_full_{args.platform}_{args.precision}_epoch{epoch}_summary.json"
        if checkpoint_path.exists() or summary_path.exists():
            raise FileExistsError(f"epoch {epoch} outputs already exist in {args.output_dir}")
        order = list(range(len(train_samples)))
        random.Random(seed + epoch).shuffle(order)
        model.train()
        loss_sum = 0.0
        min_loss = math.inf
        max_loss = -math.inf
        steps = 0
        reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        for start in range(0, len(order), args.batch_size):
            selected = [
                train_samples[index] for index in order[start : start + args.batch_size]
            ]
            batch = make_full_batch(selected, registry, config, device)
            optimizer.zero_grad(set_to_none=True)
            context = (
                nullcontext() if args.precision == "fp32"
                else torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            )
            with context:
                output = model(**batch)
            loss = float(output.loss.detach())
            if not math.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch={epoch} step={steps + 1}")
            output.loss.backward()
            optimizer.step()
            loss_sum += loss
            min_loss = min(min_loss, loss)
            max_loss = max(max_loss, loss)
            steps += 1
            if steps % 100 == 0:
                print(f"stage=train epoch={epoch} steps={steps} mean_loss={loss_sum / steps:.6f}", flush=True)
        synchronize(device)
        elapsed = time.perf_counter() - started
        global_steps += steps

        saved = {
            "variant": "full",
            "model_config": asdict(config),
            "model_state": cpu_tree(model.state_dict()),
            "optimizer_state": cpu_tree(optimizer.state_dict()),
            "sid_registry_version": registry.version,
            "boundaries": asdict(boundaries),
            "args": original_args,
            "full_training": {
                "completed_epochs": epoch,
                "global_steps": global_steps,
                "precision": args.precision,
                "batch_size": args.batch_size,
                "train_samples": len(train_samples),
                "max_train_samples": args.max_train_samples,
                "learning_rate": learning_rate,
                "source_commit": source["commit"],
                "events_sha256": input_hashes["events_sha256"],
                "sid_registry_sha256": input_hashes["sid_registry_sha256"],
            },
        }
        torch.save(saved, checkpoint_path)
        summary = {
            "schema_version": 1,
            "protocol": "oxygenrec_v2_full_epoch_training_v1",
            "platform": args.platform,
            "device": str(device),
            "device_name": device_name(device),
            "precision": args.precision,
            "runtime": {
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
                "torch_npu": package_version("torch-npu"),
            },
            "source": source,
            "inputs": input_hashes,
            "training": {
                "epoch": epoch,
                "global_steps": global_steps,
                "epoch_steps": steps,
                "train_samples": len(train_samples),
                "max_train_samples": args.max_train_samples,
                "batch_size": args.batch_size,
                "learning_rate": learning_rate,
                "optimizer_state_loaded": optimizer_loaded,
                "mean_loss": loss_sum / steps,
                "min_loss": min_loss,
                "max_loss": max_loss,
                "all_losses_finite": True,
                "behavior_counts": dict(sorted(behavior_counts.items())),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": file_sha256(checkpoint_path),
            },
            "performance": {
                "scope": "end_to_end_epoch_observation_not_formal_benchmark",
                "elapsed_seconds": elapsed,
                "steps_per_second": steps / elapsed,
                "samples_per_second": len(train_samples) / elapsed,
                "peak_memory_allocated_bytes": max_memory_allocated(device),
            },
            "status": "passed",
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"OK stage=v2_full_epoch_training platform={args.platform} "
            f"precision={args.precision} epoch={epoch} steps={steps} "
            f"mean_loss={loss_sum / steps:.6f} samples_per_second={len(train_samples) / elapsed:.3f} "
            f"checkpoint={checkpoint_path} summary={summary_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
