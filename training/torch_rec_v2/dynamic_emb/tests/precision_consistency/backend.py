#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Backend abstraction for the DynamicEmb cross-backend consistency suite.

The same test code must run on two *different* packages that are supposed to be
API compatible:

* Ascend NPU  -> ``dynamic_emb``  (this repository)
* NVIDIA GPU  -> ``dynamicemb``   (recsys-examples v25.09)

They expose the same *logical* API but not always the same import path, so tests
resolve every API through :func:`api` instead of hard-coding imports.  The
backend is selected with ``DYNAMICEMB_BACKEND`` (``npu`` | ``gpu``); when unset
it is auto-detected from whether ``torch_npu`` is importable.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, List, Optional

import torch

# --------------------------------------------------------------------------- #
# Backend detection
# --------------------------------------------------------------------------- #


def _detect_backend() -> str:
    forced = os.environ.get("DYNAMICEMB_BACKEND", "").strip().lower()
    if forced in ("npu", "gpu"):
        return forced
    if forced:
        raise ValueError(f"DYNAMICEMB_BACKEND must be 'npu' or 'gpu', got {forced!r}")
    try:
        importlib.import_module("torch_npu")  # noqa: F401

        return "npu"
    except Exception:
        return "gpu"


BACKEND: str = _detect_backend()
IS_NPU: bool = BACKEND == "npu"

#: Device type string used to build tensors ("npu" or "cuda").
DEVICE_TYPE: str = "npu" if IS_NPU else "cuda"


def device(ordinal: int = 0) -> torch.device:
    """Return the accelerator device to use."""
    return torch.device(f"{DEVICE_TYPE}:{ordinal}")


def dist_backend() -> str:
    return "hccl" if IS_NPU else "nccl"


# --------------------------------------------------------------------------- #
# API resolution (import-path agnostic)
# --------------------------------------------------------------------------- #

#: logical API name -> ordered candidate import locations ("module" or
#: "module: attr").  Both the NPU layout (everything under ``distributed``) and
#: the NVIDIA layout (``planner`` / ``shard`` subpackages) are listed so a single
#: test file can address either package.
_IMPORT_CANDIDATES: dict[str, List[str]] = {
    # --- config / enums / dataclasses -------------------------------------
    "DynamicEmbCheckMode": [
        "dynamic_emb.DynamicEmbCheckMode",
        "dynamic_emb.distributed.dynamicemb_config.DynamicEmbCheckMode",
        "dynamicemb.DynamicEmbCheckMode",
        "dynamicemb.dynamicemb_config.DynamicEmbCheckMode",
    ],
    "DynamicEmbInitializerMode": [
        "dynamic_emb.DynamicEmbInitializerMode",
        "dynamic_emb.distributed.dynamicemb_config.DynamicEmbInitializerMode",
        "dynamicemb.DynamicEmbInitializerMode",
        "dynamicemb.dynamicemb_config.DynamicEmbInitializerMode",
    ],
    "DynamicEmbInitializerArgs": [
        "dynamic_emb.DynamicEmbInitializerArgs",
        "dynamic_emb.distributed.dynamicemb_config.DynamicEmbInitializerArgs",
        "dynamicemb.DynamicEmbInitializerArgs",
        "dynamicemb.dynamicemb_config.DynamicEmbInitializerArgs",
    ],
    "DynamicEmbScoreStrategy": [
        "dynamic_emb.DynamicEmbScoreStrategy",
        "dynamic_emb.distributed.dynamicemb_config.DynamicEmbScoreStrategy",
        "dynamicemb.DynamicEmbScoreStrategy",
        "dynamicemb.dynamicemb_config.DynamicEmbScoreStrategy",
    ],
    "DynamicEmbTableOptions": [
        "dynamic_emb.DynamicEmbTableOptions",
        "dynamic_emb.distributed.dynamicemb_config.DynamicEmbTableOptions",
        "dynamicemb.DynamicEmbTableOptions",
        "dynamicemb.dynamicemb_config.DynamicEmbTableOptions",
    ],
    # --- planner / sharding ----------------------------------------------
    "DynamicEmbParameterConstraints": [
        "dynamic_emb.DynamicEmbParameterConstraints",
        "dynamic_emb.distributed.planner.DynamicEmbParameterConstraints",
        "dynamic_emb.distributed.planner.types.DynamicEmbParameterConstraints",
        "dynamicemb.planner.DynamicEmbParameterConstraints",
        "dynamicemb.planner.planner.DynamicEmbParameterConstraints",
    ],
    "DynamicEmbeddingEnumerator": [
        "dynamic_emb.DynamicEmbeddingEnumerator",
        "dynamic_emb.distributed.planner.DynamicEmbeddingEnumerator",
        "dynamic_emb.distributed.planner.enumerators.DynamicEmbeddingEnumerator",
        "dynamicemb.planner.DynamicEmbeddingEnumerator",
        "dynamicemb.planner.enumerators.DynamicEmbeddingEnumerator",
    ],
    "DynamicEmbeddingShardingPlanner": [
        "dynamic_emb.DynamicEmbeddingShardingPlanner",
        "dynamic_emb.distributed.planner.DynamicEmbeddingShardingPlanner",
        "dynamic_emb.distributed.planner.planners.DynamicEmbeddingShardingPlanner",
        "dynamicemb.planner.DynamicEmbeddingShardingPlanner",
        "dynamicemb.planner.planner.DynamicEmbeddingShardingPlanner",
    ],
    "DynamicEmbeddingBagCollectionSharder": [
        "dynamic_emb.DynamicEmbeddingBagCollectionSharder",
        "dynamic_emb.distributed.embeddingbag.DynamicEmbeddingBagCollectionSharder",
        "dynamicemb.shard.DynamicEmbeddingBagCollectionSharder",
        "dynamicemb.shard.embeddingbag.DynamicEmbeddingBagCollectionSharder",
    ],
    # --- persistence ------------------------------------------------------
    # The optimizer enum is *not* shared: dynamic_emb defines its own (str-valued)
    # EmbOptimType, while the GPU package re-exports fbgemm's.  Resolving it here
    # keeps scenario.py backend-agnostic.
    "EmbOptimType": [
        "dynamic_emb.EmbOptimType",
        "dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer.EmbOptimType",
        "fbgemm_gpu.split_embedding_configs.EmbOptimType",
    ],
    # the library's own enum -> optimizer converter (both sides raise ValueError
    # for enum members they do not implement)
    "convert_optimizer_type": [
        "dynamic_emb.distributed.optimizers.base_dynamicemb_optimizer.convert_optimizer_type",
        "dynamicemb.optimizer.convert_optimizer_type",
    ],
    "DynamicEmbDump": [
        "dynamic_emb.DynamicEmbDump",
        "dynamic_emb.distributed.dump_load.DynamicEmbDump",
        "dynamicemb.DynamicEmbDump",
        "dynamicemb.dump_load.DynamicEmbDump",
    ],
    "DynamicEmbLoad": [
        "dynamic_emb.DynamicEmbLoad",
        "dynamic_emb.distributed.dump_load.DynamicEmbLoad",
        "dynamicemb.DynamicEmbLoad",
        "dynamicemb.dump_load.DynamicEmbLoad",
    ],
}

_resolved: dict[str, Any] = {}
_unresolved: dict[str, str] = {}


def _split(spec: str) -> tuple[str, str]:
    if ":" in spec:
        mod, attr = spec.split(":", 1)
        return mod.strip(), attr.strip()
    mod, _, attr = spec.rpartition(".")
    return mod, attr


def resolve(name: str, required: bool = True) -> Optional[Any]:
    """Resolve a logical DynamicEmb API name to the object of the current backend.

    Raises :class:`ImportError` when ``required`` and nothing was found; the
    list of attempted import paths is included so the failure is actionable.
    """
    if name in _resolved:
        return _resolved[name]
    if name in _unresolved and not required:
        return None
    candidates = _IMPORT_CANDIDATES.get(name)
    if candidates is None:
        raise KeyError(f"unknown API name {name!r}; add it to _IMPORT_CANDIDATES")
    tried: list[str] = []
    for spec in candidates:
        mod_name, attr = _split(spec)
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # module not present for this backend
            tried.append(f"{mod_name}: import failed ({type(exc).__name__})")
            continue
        if hasattr(mod, attr):
            obj = getattr(mod, attr)
            _resolved[name] = obj
            return obj
        tried.append(f"{mod_name}: no attribute {attr}")
    msg = f"could not resolve '{name}' on backend={BACKEND}:\n  " + "\n  ".join(tried)
    if not required:
        _unresolved[name] = msg
        return None
    raise ImportError(msg)


def api(name: str) -> Any:
    """Shorthand for :func:`resolve` with ``required=True``."""
    return resolve(name, required=True)


def init_single_process_dist(timeout_s: int = 300) -> None:
    """Initialise a 1-rank process group so ``collective_plan``/DMP work.

    Uses ``hccl`` on NPU and ``nccl`` on GPU.  Idempotent.
    """
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", os.environ.get("DYNAMICEMB_MASTER_PORT", "29577"))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    if IS_NPU:
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
    dist.init_process_group(
        backend=dist_backend(),
        rank=0,
        world_size=1,
        timeout=__import__("datetime").timedelta(seconds=timeout_s),
    )


def destroy_dist() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def synchronize() -> None:
    if IS_NPU:
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
