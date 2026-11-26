"""Semantic-ID diagnostics corresponding to OxygenREC Section 4.1."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Sequence

from .sid import SIDRegistry


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class PrefixCoverage:
    depth: int
    occupied: int
    capacity: int
    ratio: float


@dataclass(frozen=True)
class LoadBalance:
    level: int
    p25_ratio: float
    p75_ratio: float
    p90_ratio: float


@dataclass(frozen=True)
class SIDDiagnostics:
    item_count: int
    unique_sid_count: int
    colliding_sid_count: int
    colliding_item_rate: float
    prefix_coverage: tuple[PrefixCoverage, ...]
    load_balance: tuple[LoadBalance, ...]
    collision_p90: float
    collision_p99: float
    collision_p999: float


def compute_sid_diagnostics(registry: SIDRegistry) -> SIDDiagnostics:
    """Compute coverage, collision, and per-level load-balance diagnostics."""

    paths = [sid.codes for sid in registry.item_to_sid.values()]
    item_count = len(paths)
    unique_paths = Counter(paths)
    coverage = []
    for depth in range(1, registry.levels + 1):
        occupied = len({path[:depth] for path in paths})
        capacity = registry.width**depth
        coverage.append(PrefixCoverage(depth, occupied, capacity, occupied / capacity))

    ideal_size = item_count / registry.width
    balances = []
    for level in range(registry.levels):
        counts = Counter(path[level] for path in paths)
        ratios = [counts.get(code, 0) / ideal_size for code in range(registry.width)]
        balances.append(
            LoadBalance(level + 1, _quantile(ratios, 0.25),
                        _quantile(ratios, 0.75), _quantile(ratios, 0.90))
        )

    collision_sizes = list(unique_paths.values())
    return SIDDiagnostics(
        item_count=item_count,
        unique_sid_count=len(unique_paths),
        colliding_sid_count=sum(size > 1 for size in collision_sizes),
        colliding_item_rate=registry.collision_rate(),
        prefix_coverage=tuple(coverage),
        load_balance=tuple(balances),
        collision_p90=_quantile(collision_sizes, 0.90),
        collision_p99=_quantile(collision_sizes, 0.99),
        collision_p999=_quantile(collision_sizes, 0.999),
    )
