"""Deterministic reference implementation of residual K-means.

This module prioritizes auditable semantics over throughput. It is intended for
unit tests and small experiments. Production-scale item vocabularies should use
a FAISS or accelerator-backed implementation that passes parity tests against
this reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Mapping, Sequence

from .sid import SIDRegistry, SemanticID

Vector = tuple[float, ...]


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _mean(vectors: Sequence[Vector]) -> Vector:
    return tuple(sum(values) / len(vectors) for values in zip(*vectors, strict=True))


def _validate_vectors(vectors: Sequence[Sequence[float]]) -> tuple[Vector, ...]:
    if not vectors:
        raise ValueError("vectors must not be empty")
    dimension = len(vectors[0])
    if dimension == 0:
        raise ValueError("vectors must have at least one dimension")
    normalized = tuple(tuple(float(value) for value in vector) for vector in vectors)
    if any(len(vector) != dimension for vector in normalized):
        raise ValueError("all vectors must have the same dimension")
    if any(not math.isfinite(value) for vector in normalized for value in vector):
        raise ValueError("vectors must contain only finite values")
    return normalized


@dataclass(frozen=True)
class KMeansResult:
    centroids: tuple[Vector, ...]
    assignments: tuple[int, ...]
    iterations: int
    inertia: float


class ReferenceKMeans:
    """Small deterministic Lloyd K-means with seeded K-means++ initialization."""

    def __init__(self, clusters: int, *, max_iterations: int = 100,
                 tolerance: float = 1e-6, seed: int = 0) -> None:
        if clusters < 1:
            raise ValueError("clusters must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        self.clusters = clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed

    def fit(self, vectors: Sequence[Sequence[float]]) -> KMeansResult:
        points = _validate_vectors(vectors)
        if self.clusters > len(points):
            raise ValueError(
                f"clusters ({self.clusters}) must not exceed samples ({len(points)})"
            )
        centroids = list(self._initialize(points))
        previous_assignments: tuple[int, ...] | None = None

        for iteration in range(1, self.max_iterations + 1):
            assignments = tuple(
                min(range(self.clusters),
                    key=lambda index: (_squared_distance(point, centroids[index]), index))
                for point in points
            )
            groups: list[list[Vector]] = [[] for _ in range(self.clusters)]
            for point, assignment in zip(points, assignments, strict=True):
                groups[assignment].append(point)
            updated = [
                _mean(group) if group else centroids[index]
                for index, group in enumerate(groups)
            ]
            max_shift = max(
                _squared_distance(before, after)
                for before, after in zip(centroids, updated, strict=True)
            )
            centroids = updated
            if assignments == previous_assignments or max_shift <= self.tolerance**2:
                break
            previous_assignments = assignments

        final_assignments = tuple(
            min(range(self.clusters),
                key=lambda index: (_squared_distance(point, centroids[index]), index))
            for point in points
        )
        inertia = sum(
            _squared_distance(point, centroids[assignment])
            for point, assignment in zip(points, final_assignments, strict=True)
        )
        return KMeansResult(tuple(centroids), final_assignments, iteration, inertia)

    def _initialize(self, points: tuple[Vector, ...]) -> tuple[Vector, ...]:
        generator = random.Random(self.seed)
        selected_indices = [generator.randrange(len(points))]
        while len(selected_indices) < self.clusters:
            distances = [
                min(_squared_distance(point, points[index]) for index in selected_indices)
                for point in points
            ]
            total = sum(distances)
            if total == 0:
                next_index = next(
                    index for index in range(len(points)) if index not in selected_indices
                )
            else:
                threshold = generator.random() * total
                cumulative = 0.0
                next_index = len(points) - 1
                for index, distance in enumerate(distances):
                    cumulative += distance
                    if cumulative >= threshold:
                        next_index = index
                        break
                if next_index in selected_indices:
                    next_index = next(
                        index for index in range(len(points)) if index not in selected_indices
                    )
            selected_indices.append(next_index)
        return tuple(points[index] for index in selected_indices)


@dataclass(frozen=True)
class ResidualKMeansModel:
    """Fitted codebooks for coarse-to-fine residual quantization."""

    codebooks: tuple[tuple[Vector, ...], ...]

    def __post_init__(self) -> None:
        if not self.codebooks or not self.codebooks[0] or not self.codebooks[0][0]:
            raise ValueError("codebooks must have non-zero levels, width, and dimension")
        width = len(self.codebooks[0])
        dimension = len(self.codebooks[0][0])
        for level, codebook in enumerate(self.codebooks):
            if len(codebook) != width:
                raise ValueError(f"codebook width differs at level {level}")
            if any(len(centroid) != dimension for centroid in codebook):
                raise ValueError(f"centroid dimension differs at level {level}")
            if any(
                not math.isfinite(value)
                for centroid in codebook
                for value in centroid
            ):
                raise ValueError("codebooks must contain only finite values")

    @property
    def levels(self) -> int:
        return len(self.codebooks)

    @property
    def width(self) -> int:
        return len(self.codebooks[0])

    @property
    def dimension(self) -> int:
        return len(self.codebooks[0][0])

    def encode(self, vectors: Sequence[Sequence[float]]) -> tuple[SemanticID, ...]:
        points = _validate_vectors(vectors)
        if len(points[0]) != self.dimension:
            raise ValueError(
                f"expected vectors with dimension {self.dimension}, got {len(points[0])}"
            )
        outputs: list[SemanticID] = []
        for point in points:
            residual = point
            codes = []
            for codebook in self.codebooks:
                code = min(
                    range(len(codebook)),
                    key=lambda index: (_squared_distance(residual, codebook[index]), index),
                )
                codes.append(code)
                residual = tuple(
                    value - centroid
                    for value, centroid in zip(residual, codebook[code], strict=True)
                )
            outputs.append(SemanticID(codes, levels=self.levels, width=self.width))
        return tuple(outputs)

    def reconstruct(self, semantic_ids: Sequence[SemanticID]) -> tuple[Vector, ...]:
        reconstructions = []
        for sid in semantic_ids:
            if len(sid) != self.levels:
                raise ValueError("SID level count does not match the fitted model")
            vector = [0.0] * self.dimension
            for level, code in enumerate(sid):
                if code >= len(self.codebooks[level]):
                    raise ValueError(f"SID code {code} is invalid at level {level}")
                for dimension, value in enumerate(self.codebooks[level][code]):
                    vector[dimension] += value
            reconstructions.append(tuple(vector))
        return tuple(reconstructions)

    def registry_for(self, item_embeddings: Mapping[str, Sequence[float]],
                     *, version: str) -> SIDRegistry:
        item_ids = sorted(str(item_id) for item_id in item_embeddings)
        semantic_ids = self.encode([item_embeddings[item_id] for item_id in item_ids])
        return SIDRegistry(
            dict(zip(item_ids, semantic_ids, strict=True)),
            levels=self.levels,
            width=self.width,
            version=version,
        )

    def to_json(self, path: str | Path, *, version: str) -> None:
        """Persist codebooks with shape metadata and an explicit version."""

        if not version:
            raise ValueError("version must not be empty")
        payload = {
            "version": version,
            "levels": self.levels,
            "width": self.width,
            "dimension": self.dimension,
            "codebooks": self.codebooks,
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_json(
        cls, path: str | Path, *, expected_version: str | None = None
    ) -> tuple[str, "ResidualKMeansModel"]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = str(payload["version"])
        if expected_version is not None and version != expected_version:
            raise ValueError(
                f"codebook version mismatch: expected {expected_version!r}, got {version!r}"
            )
        codebooks = tuple(
            tuple(tuple(float(value) for value in centroid) for centroid in codebook)
            for codebook in payload["codebooks"]
        )
        model = cls(codebooks)
        declared_shape = (
            int(payload["levels"]),
            int(payload["width"]),
            int(payload["dimension"]),
        )
        actual_shape = (model.levels, model.width, model.dimension)
        if actual_shape != declared_shape:
            raise ValueError(
                f"codebook shape mismatch: declared {declared_shape}, got {actual_shape}"
            )
        return version, model


class ReferenceResidualKMeans:
    """Fit one K-means codebook per residual level."""

    def __init__(self, *, levels: int = 3, width: int = 8192,
                 max_iterations: int = 100, tolerance: float = 1e-6,
                 seed: int = 0) -> None:
        if levels < 1:
            raise ValueError("levels must be positive")
        self.levels = levels
        self.width = width
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed

    def fit(self, vectors: Sequence[Sequence[float]]) -> ResidualKMeansModel:
        residuals = _validate_vectors(vectors)
        codebooks: list[tuple[Vector, ...]] = []
        for level in range(self.levels):
            result = ReferenceKMeans(
                self.width,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
                seed=self.seed + level,
            ).fit(residuals)
            codebooks.append(result.centroids)
            residuals = tuple(
                tuple(
                    value - centroid
                    for value, centroid in zip(
                        residual, result.centroids[assignment], strict=True
                    )
                )
                for residual, assignment in zip(residuals, result.assignments, strict=True)
            )
        return ResidualKMeansModel(tuple(codebooks))
