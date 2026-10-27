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

"""Deterministic train + dump/load scenario, one run per optimizer.

Written against the v25.09 Python API resolved through :mod:`backend`, so the
same code runs on Ascend ``dynamic_emb`` and on NVIDIA ``dynamicemb``.
Randomness is removed (key-derived DEBUG init, STEP score, fixed LCG inputs, no eviction),
which makes the pooled outputs a pure function of the optimizer arithmetic.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import torch

import backend as be


# Four implementations back five EmbOptimType values
# (see optimizers/base_dynamicemb_optimizer.py::convert_optimizer_type); anything
# else raises "Not supported optimizer type" at table creation and is absent."""

OPTIMIZER_SPECS: Dict[str, Dict[str, Any]] = {
    "SGD": {"family": "SGD", "kwargs": {"lr": 0.1}},
    "EXACT_SGD": {"family": "SGD", "kwargs": {"lr": 0.1}},
    "ADAM": {
        "family": "Adam",
        "kwargs": {"lr": 0.1, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0},
    },
    "EXACT_ADAGRAD": {
        "family": "AdaGrad",
        "kwargs": {"lr": 0.1, "eps": 1e-8, "initial_accumulator_value": 0.0},
    },
    "EXACT_ROWWISE_ADAGRAD": {
        "family": "RowWiseAdaGrad",
        "kwargs": {"lr": 0.1, "eps": 1e-8, "initial_accumulator_value": 0.0},
    },
}

#: default matrix -- every supported value, in a stable order
DEFAULT_OPTIMIZERS: List[str] = list(OPTIMIZER_SPECS)


# deterministic input generation


def _feature_values(seed: int, count: int, num_embeddings: int) -> List[int]:
    """Deterministic small integers (LCG) -- no dependency on torch RNG."""
    vals = []
    x = seed & 0x7FFFFFFF
    for _ in range(count):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        vals.append(x % num_embeddings)
    return vals


def build_case(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build the model + optimizer. Returns a dict of live objects."""
    import torch.distributed as dist
    import torchrec
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.distributed.planner import Topology
    from torchrec.distributed.types import ShardingType
    from fbgemm_gpu.split_embedding_configs import SparseType

    DynamicEmbParameterConstraints = be.api("DynamicEmbParameterConstraints")
    DynamicEmbeddingShardingPlanner = be.api("DynamicEmbeddingShardingPlanner")
    DynamicEmbeddingEnumerator = be.api("DynamicEmbeddingEnumerator")
    DynamicEmbeddingBagCollectionSharder = be.api("DynamicEmbeddingBagCollectionSharder")
    DynamicEmbTableOptions = be.api("DynamicEmbTableOptions")
    DynamicEmbInitializerArgs = be.api("DynamicEmbInitializerArgs")
    DynamicEmbInitializerMode = be.api("DynamicEmbInitializerMode")
    DynamicEmbScoreStrategy = be.api("DynamicEmbScoreStrategy")
    DynamicEmbCheckMode = be.api("DynamicEmbCheckMode")
    EmbOptimType = be.api("EmbOptimType")  # NPU: dynamic_emb's own enum, GPU: fbgemm's

    device = be.device()
    num_tables = cfg.get("num_tables", 2)
    embedding_dim = cfg.get("embedding_dim", 8)
    num_embeddings = cfg.get("num_embeddings", 1024)

    table_names = [f"t_{i}" for i in range(num_tables)]
    feature_names = [f"cate_{i}" for i in range(num_tables)]

    eb_configs = [
        torchrec.EmbeddingBagConfig(
            name=table_names[i],
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            feature_names=[feature_names[i]],
        )
        for i in range(num_tables)
    ]
    ebc = torchrec.EmbeddingBagCollection(device=torch.device("meta"), tables=eb_configs)

    constraints = {}
    for i in range(num_tables):
        constraints[table_names[i]] = DynamicEmbParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            compute_kernels=["fused"],
            use_dynamicemb=True,
            dynamicemb_options=DynamicEmbTableOptions(
                training=True,
                caching=bool(cfg.get("caching", False)),
                # large enough that only power-of-two rounding matters, no rehash
                init_capacity=int(cfg.get("init_capacity", 2**16)),
                max_load_factor=float(cfg.get("max_load_factor", 0.5)),
                # STEP => score is a counter (deterministic across backends)
                score_strategy=getattr(DynamicEmbScoreStrategy, str(cfg.get("score_strategy", "STEP"))),
                bucket_capacity=128,
                safe_check_mode=DynamicEmbCheckMode.IGNORE,
                global_hbm_for_values=int(cfg.get("global_hbm_for_values", 1024**3)),
                # DEBUG => every key gets its own value (key % 100000), so a
                # constant-everywhere table cannot mask bugs, and the values stay
                # bit-identical across CUDA and Ascend (the RNG modes are not).
                initializer_args=DynamicEmbInitializerArgs(
                    mode=getattr(DynamicEmbInitializerMode, str(cfg.get("initializer_mode", "DEBUG"))),
                    mean=float(cfg.get("init_mean", 0.0)),
                    std_dev=float(cfg.get("init_std_dev", 1.0)),
                    lower=cfg.get("init_lower", None),
                    upper=cfg.get("init_upper", None),
                    value=float(cfg.get("init_value", 0.5)),
                ),
                external_storage=cfg.get("external_storage", None),
            ),
        )

    topology = Topology(world_size=dist.get_world_size(), compute_device=be.DEVICE_TYPE)
    batch_size = int(cfg.get("batch_size", 8))
    enumerator = DynamicEmbeddingEnumerator(topology=topology, batch_size=batch_size, constraints=constraints)

    planner = DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        batch_size=batch_size,
        enumerator=enumerator,
        constraints=constraints,
        debug=True,
    )

    opt_name = str(cfg.get("optimizer", "EXACT_SGD"))
    if opt_name not in OPTIMIZER_SPECS:
        raise ValueError(f"unsupported optimizer {opt_name!r}; expected one of {DEFAULT_OPTIMIZERS}")
    if not hasattr(EmbOptimType, opt_name):
        raise ValueError(f"{be.BACKEND} package has no EmbOptimType.{opt_name}")
    opt_kwargs = dict(OPTIMIZER_SPECS[opt_name]["kwargs"])
    opt_kwargs.update(cfg.get("optimizer_kwargs", {}))
    fused_params = {
        "output_dtype": SparseType.FP32,
        "optimizer": getattr(EmbOptimType, opt_name),
        # fbgemm fused optimizers read the learning rate from this key
        "learning_rate": float(opt_kwargs.get("lr", 0.1)),
    }
    for _k, _v in opt_kwargs.items():
        if _k != "lr":
            fused_params[_k] = _v
    sharder = DynamicEmbeddingBagCollectionSharder(fused_params=fused_params)

    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)
    model = DistributedModelParallel(module=ebc, device=device, sharders=[sharder], plan=plan)
    model.train()

    return {
        "model": model,
        "ebc": ebc,
        "eb_configs": eb_configs,
        "table_names": table_names,
        "feature_names": feature_names,
        "embedding_dim": embedding_dim,
        "num_embeddings": num_embeddings,
        "batch_size": batch_size,
        "device": device,
    }


def make_kjt(case: Dict[str, Any], values_per_feature: List[List[int]]):
    """Build a KeyedJaggedTensor: one bag per (feature, batch-item)."""
    import torchrec

    keys: List[str] = []
    lengths: List[int] = []
    values: List[int] = []
    for fname, bags in zip(case["feature_names"], values_per_feature):
        keys.append(fname)
        for bag in bags:
            lengths.append(len(bag))
            values.extend(bag)
    return torchrec.KeyedJaggedTensor(
        keys=keys,
        values=torch.tensor(values, dtype=torch.int64, device=case["device"]),
        lengths=torch.tensor(lengths, dtype=torch.int64, device=case["device"]),
    )


def _tensor_digest(t: torch.Tensor) -> Dict[str, Any]:
    tf = t.detach().to(torch.float64).cpu()
    return {
        "shape": list(t.shape),
        "sum": float(tf.sum().item()),
        "mean": float(tf.mean().item()) if tf.numel() else 0.0,
        "abs_sum": float(tf.abs().sum().item()),
        "first8": [float(x) for x in tf.reshape(-1)[:8].tolist()],
    }


def run_scenario(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the full scenario and return a JSON-able result bundle."""
    cfg = dict(cfg or {})
    result: Dict[str, Any] = {"backend": be.BACKEND, "phases": {}}
    opt_name = str(cfg.get("optimizer", "EXACT_SGD"))
    result["optimizer"] = opt_name
    result["optimizer_family"] = OPTIMIZER_SPECS[opt_name]["family"]
    result["optimizer_kwargs"] = dict(OPTIMIZER_SPECS[opt_name]["kwargs"])
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        be.init_single_process_dist()

    rounds = int(cfg.get("rounds", 5))

    tmpdir = tempfile.mkdtemp(prefix=f"dynemb_consistency_{be.BACKEND}_")
    try:
        case = build_case(cfg)
        model = case["model"]
        result["world_size"] = dist.get_world_size()
        result["device"] = str(case["device"])

        # training batch, and a probe that is a *subset* of it so that the probe
        # output actually reflects the trained embeddings.
        train_bags = [
            [[int(v) % case["num_embeddings"] for v in bag] for bag in feature_bags]
            for feature_bags in _batches(cfg, case)
        ]
        probe_bags = [[feature_bags[0]] for feature_bags in train_bags]

        # --- initial (pre-training) probe -------------------------------- #
        with torch.no_grad():
            out0 = model(make_kjt(case, probe_bags))
        result["phases"]["initial_probe"] = _tensor_digest(out0.values())

        # --- multi-round training --------------------------------------- #
        # Squared loss: its gradient (2 * pooled output) changes as the table
        # moves, so optimizers with different update rules diverge.  A plain sum
        # has a constant unit gradient, under which Adam/AdaGrad are numerically
        # indistinguishable from SGD -- useless for a per-optimizer precision test.
        loss_mode = str(cfg.get("loss_mode", "sq"))
        round_digests = []
        for r in range(rounds):
            kjt = make_kjt(case, train_bags)
            out = model(kjt)
            pooled = out.values().float()
            loss = (pooled * pooled).sum() if loss_mode == "sq" else pooled.sum()
            loss.backward()
            be.synchronize()
            round_digests.append(
                {"round": r, "loss": float(loss.detach().cpu().item()), "out": _tensor_digest(out.values())}
            )
        result["phases"]["training"] = round_digests

        # --- post-training probe ---------------------------------------- #
        with torch.no_grad():
            out1 = model(make_kjt(case, probe_bags))
        result["phases"]["post_training_probe"] = _tensor_digest(out1.values())

        # --- dump / load roundtrip -------------------------------------- #
        # Dump the trained model, load into a *fresh* model and check the fresh
        # model now reproduces the trained probe output.
        dump_dir = os.path.join(tmpdir, "dump")
        os.makedirs(dump_dir, exist_ok=True)
        DynamicEmbDump = be.api("DynamicEmbDump")
        DynamicEmbLoad = be.api("DynamicEmbLoad")
        trained_probe = out1.values().detach().to(torch.float32).cpu()
        DynamicEmbDump(dump_dir, model, optim=True)
        be.barrier()

        case2 = build_case(cfg)
        fresh_before = _tensor_probe_raw(case2["model"], case2, probe_bags)
        DynamicEmbLoad(dump_dir, case2["model"], optim=True)
        be.barrier()
        after_load = _tensor_probe_raw(case2["model"], case2, probe_bags)
        result["phases"]["dump_load"] = {
            "fresh_before_load": _tensor_digest(fresh_before),
            "after_load": _tensor_digest(after_load),
            "load_changed_state": not _close(fresh_before, after_load),
            "loaded_matches_trained": _close(after_load, trained_probe),
        }
        result["_tensors"] = {
            "post_training_probe": trained_probe,
            "after_load_probe": after_load,
        }

    except Exception as exc:  # noqa: BLE001
        result["fatal"] = {"error_type": type(exc).__name__, "detail": str(exc)[:1000]}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        # the process group stays alive: callers run several optimizers in one
        # process and tear it down once (see test module-level fixture).
    return result


# small helpers


def _batches(cfg: Dict[str, Any], case: Dict[str, Any]) -> List[List[int]]:
    n = case["batch_size"]
    per_feature = []
    for i in range(len(case["feature_names"])):
        vals = _feature_values(1234 + i, n * 3, case["num_embeddings"])
        per_feature.append([vals[j * 3 : j * 3 + 3] for j in range(n)])
    return per_feature


def _tensor_probe_raw(model, case, probe_bags) -> torch.Tensor:
    with torch.no_grad():
        return model(make_kjt(case, probe_bags)).values().detach().to(torch.float32).cpu()


def _close(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    return bool(torch.allclose(a, b, rtol=rtol, atol=atol))


def compact_result(res: Dict[str, Any]) -> Dict[str, Any]:
    """Project a scenario result onto the fields the precision cases compare."""
    phases = res.get("phases", {})
    keys = ("sum", "mean", "abs_sum")
    dump_load = phases.get("dump_load", {})
    return {
        "losses": [round_["loss"] for round_ in phases.get("training", [])],
        "initial_probe": {k: phases.get("initial_probe", {}).get(k) for k in keys},
        "post_training_probe": {k: phases.get("post_training_probe", {}).get(k) for k in keys},
        "dump_load": {k: dump_load.get(k) for k in ("load_changed_state", "loaded_matches_trained")},
    }
