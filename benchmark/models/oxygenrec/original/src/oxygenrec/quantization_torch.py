"""Chunked PyTorch residual K-Means for accelerator-scale SID fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .sid import SIDRegistry


def _validate_vectors(vectors: Tensor, width: int) -> Tensor:
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("vectors must have shape [items, dimension]")
    if vectors.shape[0] < width:
        raise ValueError("item count must be at least the codebook width")
    if not vectors.is_floating_point():
        raise ValueError("vectors must use a floating-point dtype")
    if not torch.isfinite(vectors).all():
        raise ValueError("vectors must contain only finite values")
    return vectors.detach().to(dtype=torch.float32)


def _assign(vectors: Tensor, centroids: Tensor, chunk_size: int) -> Tensor:
    assignments = []
    centroid_norm = (centroids * centroids).sum(dim=1).unsqueeze(0)
    for start in range(0, vectors.shape[0], chunk_size):
        chunk = vectors[start : start + chunk_size]
        distances = (
            (chunk * chunk).sum(dim=1, keepdim=True)
            + centroid_norm
            - 2.0 * chunk @ centroids.transpose(0, 1)
        )
        assignments.append(distances.argmin(dim=1))
    return torch.cat(assignments)


@dataclass(frozen=True)
class TorchResidualKMeansModel:
    """Dense tensor codebooks with chunked encode and explicit persistence."""

    codebooks: Tensor  # [levels, width, dimension]

    def __post_init__(self) -> None:
        if self.codebooks.ndim != 3 or min(self.codebooks.shape) < 1:
            raise ValueError("codebooks must have shape [levels, width, dimension]")
        if not self.codebooks.is_floating_point():
            raise ValueError("codebooks must use a floating-point dtype")
        if not torch.isfinite(self.codebooks).all():
            raise ValueError("codebooks must contain only finite values")

    @property
    def levels(self) -> int:
        return self.codebooks.shape[0]

    @property
    def width(self) -> int:
        return self.codebooks.shape[1]

    @property
    def dimension(self) -> int:
        return self.codebooks.shape[2]

    def encode(self, vectors: Tensor, *, chunk_size: int = 4096) -> Tensor:
        points = _validate_vectors(vectors, self.width)
        if points.shape[1] != self.dimension:
            raise ValueError(
                f"expected dimension {self.dimension}, got {points.shape[1]}"
            )
        points = points.to(self.codebooks.device)
        residuals = points
        codes = []
        for codebook in self.codebooks:
            assignments = _assign(residuals, codebook, chunk_size)
            codes.append(assignments)
            residuals = residuals - codebook[assignments]
        return torch.stack(codes, dim=1)

    def reconstruct(self, codes: Tensor) -> Tensor:
        if codes.ndim != 2 or codes.shape[1] != self.levels:
            raise ValueError("codes must have shape [items, levels]")
        if codes.dtype != torch.long:
            raise ValueError("codes must use torch.long")
        if (codes < 0).any() or (codes >= self.width).any():
            raise ValueError("codes contain an index outside the codebook")
        codes = codes.to(self.codebooks.device)
        output = torch.zeros(
            codes.shape[0],
            self.dimension,
            dtype=self.codebooks.dtype,
            device=self.codebooks.device,
        )
        for level, codebook in enumerate(self.codebooks):
            output += codebook[codes[:, level]]
        return output

    def registry_for(
        self,
        item_ids: Sequence[str],
        vectors: Tensor,
        *,
        version: str,
        chunk_size: int = 4096,
    ) -> SIDRegistry:
        if len(item_ids) != vectors.shape[0]:
            raise ValueError("item_ids and vectors must have equal length")
        if len(set(str(item_id) for item_id in item_ids)) != len(item_ids):
            raise ValueError("item_ids must be unique")
        codes = self.encode(vectors, chunk_size=chunk_size).cpu().tolist()
        return SIDRegistry(
            dict(zip((str(item) for item in item_ids), codes, strict=True)),
            levels=self.levels,
            width=self.width,
            version=version,
        )

    def save(self, path: str | Path, *, version: str) -> None:
        if not version:
            raise ValueError("version must not be empty")
        torch.save(
            {
                "schema_version": 1,
                "version": version,
                "codebooks": self.codebooks.detach().cpu(),
            },
            Path(path),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        expected_version: str | None = None,
    ) -> tuple[str, "TorchResidualKMeansModel"]:
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported torch codebook schema version")
        version = str(payload["version"])
        if expected_version is not None and version != expected_version:
            raise ValueError(
                f"codebook version mismatch: expected {expected_version!r}, got {version!r}"
            )
        return version, cls(payload["codebooks"].to(device))


class TorchResidualKMeans:
    """Fit one chunked Lloyd K-Means codebook per residual level."""

    def __init__(
        self,
        *,
        levels: int = 3,
        width: int = 8192,
        max_iterations: int = 25,
        tolerance: float = 1e-4,
        seed: int = 0,
        assignment_chunk_size: int = 4096,
        initialization: str = "random",
    ) -> None:
        if levels < 1 or width < 1 or max_iterations < 1:
            raise ValueError("levels, width, and max_iterations must be positive")
        if tolerance < 0 or assignment_chunk_size < 1:
            raise ValueError("tolerance must be non-negative and chunk size positive")
        self.levels = levels
        self.width = width
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed
        self.assignment_chunk_size = assignment_chunk_size
        if initialization not in {"random", "kmeans++"}:
            raise ValueError("initialization must be 'random' or 'kmeans++'")
        self.initialization = initialization

    def _initialize(self, vectors: Tensor, *, level: int) -> Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + level)
        if self.initialization == "random":
            indices = torch.randperm(
                vectors.shape[0], generator=generator, device="cpu"
            )[: self.width]
            return vectors[indices.to(vectors.device)].clone()

        first = int(torch.randint(vectors.shape[0], (1,), generator=generator))
        selected = [first]
        closest = (vectors - vectors[first]).square().sum(dim=1)
        for _ in range(1, self.width):
            total = float(closest.sum())
            if total <= 0.0:
                selected_set = set(selected)
                next_index = next(
                    index for index in range(vectors.shape[0])
                    if index not in selected_set
                )
            else:
                next_index = int(
                    torch.multinomial(
                        closest.detach().cpu(), 1, generator=generator
                    ).item()
                )
            selected.append(next_index)
            distance = (vectors - vectors[next_index]).square().sum(dim=1)
            closest = torch.minimum(closest, distance)
        indices = torch.tensor(selected, dtype=torch.long, device=vectors.device)
        return vectors[indices].clone()

    def fit(self, vectors: Tensor) -> TorchResidualKMeansModel:
        residuals = _validate_vectors(vectors, self.width)
        device = vectors.device
        residuals = residuals.to(device)
        codebooks = []
        for level in range(self.levels):
            centroids = self._initialize(residuals, level=level)
            for _ in range(self.max_iterations):
                assignments = _assign(
                    residuals, centroids, self.assignment_chunk_size
                )
                sums = torch.zeros_like(centroids)
                sums.index_add_(0, assignments, residuals)
                counts = torch.bincount(
                    assignments, minlength=self.width
                ).to(dtype=centroids.dtype)
                updated = torch.where(
                    counts[:, None] > 0,
                    sums / counts.clamp_min(1)[:, None],
                    centroids,
                )
                shift = (updated - centroids).square().sum(dim=1).max()
                centroids = updated
                if float(shift) <= self.tolerance**2:
                    break
            assignments = _assign(
                residuals, centroids, self.assignment_chunk_size
            )
            codebooks.append(centroids)
            residuals = residuals - centroids[assignments]
        return TorchResidualKMeansModel(torch.stack(codebooks))
