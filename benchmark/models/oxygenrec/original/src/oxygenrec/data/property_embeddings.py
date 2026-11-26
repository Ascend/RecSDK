"""Leak-resistant proxy item vectors from RetailRocket property snapshots."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


EXPECTED_COLUMNS = ("timestamp", "itemid", "property", "value")


@dataclass(frozen=True)
class PropertyEmbeddingResult:
    item_ids: tuple[str, ...]
    vectors: np.ndarray
    selected_item_count: int
    represented_item_count: int
    retained_snapshot_count: int
    scanned_row_count: int


def _feature_bucket(feature: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
    index = int.from_bytes(digest[:8], "little") % dimension
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


def build_property_hash_embeddings(
    property_paths: Sequence[str | Path],
    selected_item_ids: Iterable[str],
    *,
    train_end_ms: int,
    dimension: int = 256,
) -> PropertyEmbeddingResult:
    """Hash each item's latest pre-cutoff property values into a dense vector.

    Only the latest snapshot for each ``(item, property)`` pair strictly before
    ``train_end_ms`` is retained. Values are hashed immediately and never stored
    in the returned result.
    """

    if dimension < 1:
        raise ValueError("dimension must be positive")
    if train_end_ms < 1:
        raise ValueError("train_end_ms must be positive")
    selected = frozenset(str(item) for item in selected_item_ids)
    if not selected:
        raise ValueError("selected_item_ids must not be empty")
    latest: dict[tuple[str, str], tuple[int, int, float]] = {}
    scanned = 0
    for raw_path in property_paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
                raise ValueError(
                    f"unexpected property columns in {path.name}: {reader.fieldnames}"
                )
            for row_number, row in enumerate(reader, start=2):
                scanned += 1
                item_id = row["itemid"].strip()
                if item_id not in selected:
                    continue
                try:
                    timestamp = int(row["timestamp"])
                except ValueError as error:
                    raise ValueError(
                        f"invalid timestamp in {path.name} row {row_number}"
                    ) from error
                if timestamp >= train_end_ms:
                    continue
                property_name = row["property"].strip()
                value = row["value"].strip()
                if not property_name or not value:
                    continue
                key = (item_id, property_name)
                previous = latest.get(key)
                if previous is None or timestamp > previous[0]:
                    index, sign = _feature_bucket(
                        f"{property_name}={value}", dimension
                    )
                    latest[key] = (timestamp, index, sign)

    represented = sorted({item_id for item_id, _ in latest})
    if not represented:
        raise ValueError("no selected items have properties before the train cutoff")
    row_for = {item_id: row for row, item_id in enumerate(represented)}
    vectors = np.zeros((len(represented), dimension), dtype=np.float32)
    for (item_id, _), (_, index, sign) in latest.items():
        vectors[row_for[item_id], index] += sign
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0
    if not np.all(nonzero):
        represented = [item for item, keep in zip(represented, nonzero, strict=True) if keep]
        vectors = vectors[nonzero]
        norms = norms[nonzero]
    vectors /= norms
    return PropertyEmbeddingResult(
        item_ids=tuple(represented),
        vectors=vectors,
        selected_item_count=len(selected),
        represented_item_count=len(represented),
        retained_snapshot_count=len(latest),
        scanned_row_count=scanned,
    )
