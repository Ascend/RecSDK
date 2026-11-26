#!/usr/bin/env python3
"""Build proxy item vectors and fit a real RetailRocket RQ SID registry."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from oxygenrec.data.events import load_retailrocket_events
from oxygenrec.data.property_embeddings import build_property_hash_embeddings
from oxygenrec.data.temporal import Split, TemporalBoundaries
from oxygenrec.quantization_torch import TorchResidualKMeans
from oxygenrec.sid_metrics import compute_sid_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=Path("data/raw/retailrocket/events.csv"))
    parser.add_argument(
        "--properties",
        type=Path,
        nargs=2,
        default=(
            Path("data/raw/retailrocket/item_properties_part1.csv"),
            Path("data/raw/retailrocket/item_properties_part2.csv"),
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/retailrocket_sid"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-items", type=int, default=5_000)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--initialization", choices=("random", "kmeans++"), default="random"
    )
    return parser.parse_args()


def event_boundaries(path: Path) -> TemporalBoundaries:
    minimum = None
    maximum = None
    for event in load_retailrocket_events(path):
        minimum = event.timestamp_ms if minimum is None else min(minimum, event.timestamp_ms)
        maximum = event.timestamp_ms if maximum is None else max(maximum, event.timestamp_ms)
    if minimum is None or maximum is None:
        raise ValueError("events.csv contains no events")
    duration = maximum - minimum + 1
    return TemporalBoundaries(
        train_end_ms=minimum + duration * 8 // 10,
        validation_end_ms=minimum + duration * 9 // 10,
    )


def select_train_items(
    path: Path, boundaries: TemporalBoundaries, max_items: int
) -> tuple[str, ...]:
    counts = Counter(
        event.item_id
        for event in load_retailrocket_events(path)
        if boundaries.split_for(event.timestamp_ms) is Split.TRAIN
    )
    return tuple(sorted(counts, key=lambda item: (-counts[item], item))[:max_items])


def main() -> int:
    args = parse_args()
    print("stage=select_train_items")
    boundaries = event_boundaries(args.events)
    selected_items = select_train_items(args.events, boundaries, args.max_items)
    print("stage=build_property_embeddings")
    embeddings = build_property_hash_embeddings(
        args.properties,
        selected_items,
        train_end_ms=boundaries.train_end_ms,
        dimension=args.dimension,
    )
    if embeddings.represented_item_count < args.width:
        raise RuntimeError("represented item count is smaller than RQ codebook width")
    device = torch.device(args.device)
    vectors = torch.from_numpy(embeddings.vectors).to(device)
    print("stage=fit_residual_kmeans")
    fitter = TorchResidualKMeans(
        levels=args.levels,
        width=args.width,
        max_iterations=args.iterations,
        seed=args.seed,
        assignment_chunk_size=args.chunk_size,
        initialization=args.initialization,
    )
    model = fitter.fit(vectors)
    unique_vector_count = int(np.unique(embeddings.vectors, axis=0).shape[0])
    unique_vector_rate = unique_vector_count / embeddings.represented_item_count
    version = (
        f"retailrocket-property-hash-d{args.dimension}-cut{boundaries.train_end_ms}"
        f"-rq-l{args.levels}-w{args.width}-s{args.seed}"
        f"-init{args.initialization}"
    )
    registry = model.registry_for(
        embeddings.item_ids,
        vectors,
        version=version,
        chunk_size=args.chunk_size,
    )
    codes = model.encode(vectors, chunk_size=args.chunk_size)
    reconstruction = model.reconstruct(codes)
    baseline_mse = float(torch.mean(vectors**2))
    reconstruction_mse = float(torch.mean((vectors - reconstruction) ** 2))
    diagnostics = compute_sid_diagnostics(registry)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "item_embeddings.npy", embeddings.vectors, allow_pickle=False)
    (args.output_dir / "item_ids.txt").write_text(
        "\n".join(embeddings.item_ids) + "\n", encoding="utf-8"
    )
    model.save(args.output_dir / "rq_codebooks.pt", version=version)
    registry.to_json(args.output_dir / "sid_registry.json")
    report = {
        "schema_version": 1,
        "version": version,
        "representation": "latest-pre-cutoff-property-feature-hashing-proxy",
        "paper_multimodal_embedding_reproduction": False,
        "boundaries": asdict(boundaries),
        "embedding": {
            "dimension": args.dimension,
            "selected_items": embeddings.selected_item_count,
            "represented_items": embeddings.represented_item_count,
            "unique_vectors": unique_vector_count,
            "unique_vector_rate": unique_vector_rate,
            "retained_property_snapshots": embeddings.retained_snapshot_count,
            "scanned_property_rows": embeddings.scanned_row_count,
        },
        "quantization": {
            "levels": args.levels,
            "width": args.width,
            "iterations": args.iterations,
            "seed": args.seed,
            "initialization": args.initialization,
            "baseline_mse": baseline_mse,
            "reconstruction_mse": reconstruction_mse,
        },
        "diagnostics": asdict(diagnostics),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"OK represented_items={embeddings.represented_item_count} "
        f"unique_vector_rate={unique_vector_rate:.6f} "
        f"shape={tuple(model.codebooks.shape)} "
        f"mse={baseline_mse:.6f}->{reconstruction_mse:.6f} "
        f"collision_rate={diagnostics.colliding_item_rate:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
