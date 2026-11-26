#!/usr/bin/env python3
"""Evaluate one frozen v2 Full checkpoint on a deterministic validation cohort."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from oxygenrec.device import device_name, resolve_device, seed_torch, synchronize
from oxygenrec.migration_alignment import (
    build_fixed_batch,
    file_sha256,
    forward_kwargs,
    generation_kwargs,
    git_state,
    load_model,
    move_batch,
    restored_config,
)
from oxygenrec.migration_validation import (
    canonical_sha256,
    summarize_listwise_validation,
)
from oxygenrec.sid import PrefixTrie, SIDRegistry


PROTOCOL = "oxygenrec_v2_full_fixed_validation_v1"
SOURCE_FILES = (
    "run_v2_validation.sh",
    "scripts/evaluate_v2_fixed_validation.py",
    "src/oxygenrec/device.py",
    "src/oxygenrec/evaluation.py",
    "src/oxygenrec/migration_alignment.py",
    "src/oxygenrec/migration_validation.py",
    "src/oxygenrec/model.py",
    "src/oxygenrec/data/model_inputs.py",
    "src/oxygenrec/data/temporal.py",
    "src/oxygenrec/sid.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("gpu", "npu"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sid-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases-output",
        type=Path,
        help="optional compact per-case target/greedy/beam output for changed-case drilldown",
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beam-width", type=int, default=5)
    return parser.parse_args()


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@contextmanager
def transformer_fastpath_disabled():
    """Use the same unfused Transformer path on both devices for correctness checks."""

    mha = getattr(torch.backends, "mha", None)
    getter = getattr(mha, "get_fastpath_enabled", None)
    setter = getattr(mha, "set_fastpath_enabled", None)
    if not callable(getter) or not callable(setter):
        raise RuntimeError("PyTorch MHA fastpath control is unavailable")
    previous = bool(getter())
    setter(False)
    try:
        yield previous
    finally:
        setter(previous)


def batch_slice(batch: dict[str, torch.Tensor], start: int, stop: int):
    return {name: value[start:stop] for name, value in batch.items()}


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.beam_width < 1:
        raise ValueError("samples, batch-size and beam-width must be positive")
    missing = [
        str(path)
        for path in (args.events, args.checkpoint, args.sid_registry)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("missing fixed-validation inputs: " + ", ".join(missing))

    device = resolve_device(args.device)
    expected_type = "cuda" if args.platform == "gpu" else "npu"
    if device.type != expected_type:
        raise ValueError(f"platform {args.platform!r} requires a {expected_type!r} device")

    project_root = Path(__file__).resolve().parents[1]
    source = git_state(project_root)
    if source["commit"] is None or source["tracked_dirty"]:
        raise RuntimeError("fixed validation requires a known commit and clean tracked files")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != "full":
        raise ValueError("fixed validation requires a v2 Full checkpoint")
    registry = SIDRegistry.from_json(args.sid_registry)
    if checkpoint.get("sid_registry_version") != registry.version:
        raise RuntimeError("checkpoint and SID registry versions differ")
    config = restored_config(checkpoint)
    cpu_batch, metadata = build_fixed_batch(
        args.events,
        checkpoint,
        registry,
        samples=args.samples,
    )
    seed = int(metadata["sample_seed"])
    seed_torch(seed, device)
    model = load_model(checkpoint, config, device)
    model.eval()
    trie = PrefixTrie.from_registry(registry)

    targets: list[list[int]] = []
    greedy: list[list[int]] = []
    beams: list[list[list[int]]] = []
    beam_scores: list[list[float]] = []
    losses: list[float] = []
    logits_finite = True
    synchronize(device)
    started = time.perf_counter()
    with transformer_fastpath_disabled() as fastpath_was_enabled:
        with torch.inference_mode():
            for start in range(0, args.samples, args.batch_size):
                stop = min(start + args.batch_size, args.samples)
                batch = move_batch(batch_slice(cpu_batch, start, stop), device)
                with autocast_context(device, args.precision):
                    output = model(**forward_kwargs(batch))
                    generated = model.generate(
                        batch["history_sids"],
                        batch["history_padding_mask"],
                        trie,
                        output_items=config.max_target_items,
                        **generation_kwargs(batch),
                    )
                    beam = model.beam_search(
                        batch["history_sids"],
                        batch["history_padding_mask"],
                        trie,
                        beam_width=args.beam_width,
                        output_items=config.max_target_items,
                        **generation_kwargs(batch),
                    )
                loss = float(output.loss.detach())
                if not math.isfinite(loss):
                    raise RuntimeError("fixed validation produced a non-finite loss")
                losses.extend([loss] * (stop - start))
                logits_finite = logits_finite and all(
                    bool(torch.isfinite(value).all()) for value in output.logits
                )
                targets.extend(
                    batch["target_sids"].reshape(stop - start, -1).cpu().tolist()
                )
                greedy.extend(generated.cpu().tolist())
                beams.extend(beam.semantic_ids.cpu().tolist())
                beam_scores.extend(beam.scores.float().cpu().tolist())
    synchronize(device)
    elapsed = time.perf_counter() - started
    if not logits_finite:
        raise RuntimeError("fixed validation produced non-finite logits")

    metrics = summarize_listwise_validation(
        targets,
        greedy,
        beams,
        sid_levels=registry.levels,
        is_legal=trie.contains,
    )
    if metrics["greedy_legal_item_rate"] != 1.0:
        raise RuntimeError("greedy decoding emitted an illegal SID")
    if metrics["beam_legal_item_rate"] != 1.0:
        raise RuntimeError("beam decoding emitted an illegal SID")

    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL,
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
        "source_file_sha256": {
            path: file_sha256(project_root / path) for path in SOURCE_FILES
        },
        "inputs": {
            "events_sha256": file_sha256(args.events),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "sid_registry_sha256": file_sha256(args.sid_registry),
            "sid_registry_version": registry.version,
        },
        "validation": {
            **metadata,
            "batch_size": args.batch_size,
            "beam_width": args.beam_width,
            "autocast_enabled": args.precision == "bf16",
            "autocast_dtype": "bfloat16" if args.precision == "bf16" else None,
            "transformer_fastpath_disabled": True,
            "transformer_fastpath_was_enabled": fastpath_was_enabled,
            "mean_loss": sum(losses) / len(losses),
            "all_logits_finite": logits_finite,
        },
        "metrics": metrics,
        "fingerprints": {
            "targets_sha256": canonical_sha256(targets),
            "greedy_sha256": canonical_sha256(greedy),
            "beam_sha256": canonical_sha256(beams),
        },
        "execution": {
            "scope": "correctness_only_not_performance_benchmark",
            "elapsed_seconds": elapsed,
        },
        "status": "passed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.cases_output is not None:
        case_payload = {
            "schema_version": 1,
            "protocol": PROTOCOL + "_cases",
            "platform": args.platform,
            "precision": args.precision,
            "source_commit": source["commit"],
            "inputs": summary["inputs"],
            "fingerprints": summary["fingerprints"],
            "cases": [
                {
                    "case_index": index,
                    "target": target,
                    "greedy": greedy_path,
                    "beam": beam_paths,
                    "beam_scores": scores,
                }
                for index, (target, greedy_path, beam_paths, scores) in enumerate(
                    zip(targets, greedy, beams, beam_scores, strict=True)
                )
            ],
        }
        args.cases_output.parent.mkdir(parents=True, exist_ok=True)
        args.cases_output.write_text(
            json.dumps(case_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    beam_hit = metrics["beam_exact_list_hit_rate"][str(args.beam_width)]
    print(
        "OK stage=v2_fixed_validation "
        f"platform={args.platform} device={device} precision={args.precision} "
        f"samples={args.samples} mean_loss={summary['validation']['mean_loss']:.6f} "
        f"sid_recall={metrics['sid_recall']:.6f} "
        f"exact_list_rate={metrics['exact_list_rate']:.6f} "
        f"beam_exact_list_hit_at_{args.beam_width}={beam_hit:.6f} "
        f"summary={args.output} cases={args.cases_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())