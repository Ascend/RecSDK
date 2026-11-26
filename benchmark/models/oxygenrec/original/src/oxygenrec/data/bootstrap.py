"""Temporary non-semantic SID bootstrap for real-data pipeline validation."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from ..sid import SIDRegistry
from .events import InteractionEvent
from .temporal import Split, TemporalBoundaries


def build_frequency_bootstrap_registry(
    events: Iterable[InteractionEvent],
    boundaries: TemporalBoundaries,
    *,
    max_items: int,
    levels: int = 3,
    width: int = 64,
) -> SIDRegistry:
    """Assign frequent train-visible items deterministic base-``width`` codes.

    This is a plumbing baseline, not a semantic tokenizer. Frequency selects
    the vocabulary; lexical item-ID order assigns stable, collision-free codes.
    """

    if max_items < 1:
        raise ValueError("max_items must be positive")
    if levels < 1 or width < 2:
        raise ValueError("levels must be positive and width must be at least 2")
    capacity = width**levels
    if max_items > capacity:
        raise ValueError(f"max_items exceeds SID capacity {capacity}")
    counts = Counter(
        event.item_id
        for event in events
        if boundaries.split_for(event.timestamp_ms) is Split.TRAIN
    )
    selected = sorted(counts, key=lambda item: (-counts[item], item))[:max_items]
    if not selected:
        raise ValueError("training split contains no items")
    stable_items = sorted(selected)
    mapping = {}
    for rank, item_id in enumerate(stable_items):
        codes = tuple(
            rank // (width ** (levels - level - 1)) % width
            for level in range(levels)
        )
        mapping[item_id] = codes
    return SIDRegistry(
        mapping,
        levels=levels,
        width=width,
        version=f"frequency-bootstrap-l{levels}-w{width}-n{len(mapping)}",
    )
