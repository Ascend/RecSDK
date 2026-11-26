"""Qwen Contextual Reasoning Instruction离线特征缓存协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from .data.temporal import NextItemSample


INSTRUCTION_CACHE_VERSION = "oxygenrec_instruction_features_v1"


def instruction_sample_key(sample: NextItemSample) -> str:
    """用split与CSV物理行号生成稳定、无用户信息的样本键。"""

    return f"{sample.split.value}:{sample.target.source_row}"


def save_instruction_feature_cache(
    path: str | Path,
    *,
    sample_keys: Sequence[str],
    features,
    metadata: Mapping[str, object],
) -> None:
    """保存连续Tensor[N,H]及其稳定样本键；默认拒绝覆盖已有缓存。"""
    import torch

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(
            f"instruction feature cache already exists: {destination}"
        )
    if features.ndim != 2 or features.shape[0] != len(sample_keys):
        raise ValueError("features must be [N,H] and match sample_keys")
    if not sample_keys or len(set(sample_keys)) != len(sample_keys):
        raise ValueError("sample_keys must be non-empty and unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": INSTRUCTION_CACHE_VERSION,
        "sample_keys": tuple(sample_keys),
        "features": features.detach().to(device="cpu"),
        "feature_size": int(features.shape[1]),
        "metadata": dict(metadata),
    }, destination)


def load_instruction_feature_cache(
    path: str | Path,
) -> tuple[object, dict[str, int], dict[str, object]]:
    """加载并校验缓存，返回Tensor[N,H]、样本键索引和元数据。"""
    import torch

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != INSTRUCTION_CACHE_VERSION:
        raise ValueError("unsupported instruction feature cache format")
    sample_keys = payload.get("sample_keys")
    features = payload.get("features")
    feature_size = payload.get("feature_size")
    metadata = payload.get("metadata")
    if not isinstance(sample_keys, (tuple, list)) or not sample_keys:
        raise ValueError("instruction cache sample_keys are missing")
    if not all(isinstance(key, str) and key for key in sample_keys):
        raise ValueError("instruction cache contains an invalid sample key")
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("instruction cache contains duplicate sample keys")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("instruction cache features must be Tensor[N,H]")
    if features.shape[0] != len(sample_keys) or features.shape[1] != feature_size:
        raise ValueError("instruction cache feature shape does not match metadata")
    if not torch.isfinite(features).all():
        raise ValueError("instruction cache contains non-finite features")
    if not isinstance(metadata, dict):
        raise ValueError("instruction cache metadata must be an object")
    return features, {key: index for index, key in enumerate(sample_keys)}, metadata
